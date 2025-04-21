from pytorch_lightning.cli import LightningCLI
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
import torch

if __name__ == '__main__':
    # Nếu không cần deterministic hoàn toàn, có thể tắt
    torch.use_deterministic_algorithms(False)

    # LightningCLI sẽ tự khởi tạo WandbLogger từ config.yaml
    cli = LightningCLI(
        LitCoMER,
        CROHMEDatamodule,
        trainer_defaults={"strategy": "auto"},
        save_config_kwargs={"overwrite": True},
    )
