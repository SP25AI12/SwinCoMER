import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        # Number of channel inputs to conv: 2*nhead if both coverages, else nhead
        in_chs = 2 * nhead if (cross_coverage and self_coverage) else nhead
        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        # Use standard BatchNorm2d instead of mask-based norm
        self.post_norm = nn.BatchNorm2d(nhead)

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor
    ) -> Tensor:
        # prev_attn: [(b * nhead), t, l]
        # key_padding_mask: [b, l], l = h*w
        # h: rows count, curr_attn same shape as prev_attn
        t = curr_attn.shape[1]
        # build spatial mask for conv/masked_fill: [b*t, 1, h, w]
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t).bool()

        # reshape attention to [b, nhead, t, l]
        curr = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev)
        if self.self_coverage:
            attns.append(curr)
        attns = torch.cat(attns, dim=1)  # [b, in_chs, t, l]

        # deterministic cumsum on CPU then back to device
        attns_cpu = attns.cpu()
        cum = attns_cpu.cumsum(dim=2)
        attns = cum.to(attns.device) - attns

        # reshape for conv: [b*t, in_chs, h, w]
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)
        cov = self.conv(attns)
        cov = self.act(cov)

        # mask out invalid spatial regions
        cov = cov.masked_fill(mask, 0.0)

        # project back to nhead channels
        cov = self.proj(cov)
        # apply standard BatchNorm2d
        cov = self.post_norm(cov)

        # reshape output: [(b*nhead), t, (h*w)]
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        return cov
