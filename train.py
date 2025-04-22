from pytorch_lightning.cli import LightningCLI
from comer.datamodule.customdatamodule import CustomDataModule
from comer.lit_comer import LitCoMER
import torch

if __name__ == '__main__':
    # Nếu không cần deterministic hoàn toàn, có thể tắt
    torch.use_deterministic_algorithms(False)

    # LightningCLI sẽ tự khởi tạo WandbLogger từ config.yaml
    cli = LightningCLI(
        LitCoMER,
        CustomDataModule,
        trainer_defaults={"strategy": "auto"},
        save_config_kwargs={"overwrite": True},
    )
