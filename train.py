from pytorch_lightning.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
import torch

if __name__ == '__main__':
    torch.use_deterministic_algorithms(False)  # Tắt tính xác định

    cli = LightningCLI(
        LitCoMER,
        CROHMEDatamodule,
        trainer_defaults={"strategy": "auto"},
        save_config_kwargs={"overwrite": True},
    )