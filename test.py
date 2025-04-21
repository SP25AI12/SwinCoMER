import os
import typer
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

def main(
    version: str = typer.Argument("0", help="Phiên bản mô hình (ví dụ: 0 cho lightning_logs/version_0)"),
    test_year: str = typer.Argument("2014", help="Năm kiểm tra CROHME (ví dụ: 2014, 2016, 2019)"),
    ckp: str = typer.Argument(
        "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt",
        help="Đường dẫn đến tệp điểm kiểm tra"
    )
):
    seed_everything(7, workers=True)

    # Kiểm tra năm hợp lệ
    valid_years = ["2014", "2016", "2019"]  # Thay bằng danh sách năm thực tế
    if test_year not in valid_years:
        raise ValueError(f"Năm kiểm tra không hợp lệ: {test_year}. Phải là một trong {valid_years}")

    # Kiểm tra tệp điểm kiểm tra
    if not os.path.exists(ckp):
        raise FileNotFoundError(
            f"Tệp điểm kiểm tra {ckp} không tồn tại. "
            "Vui lòng kiểm tra đường dẫn hoặc chạy mã huấn luyện để tạo tệp .ckpt."
        )

    trainer = Trainer(logger=False, accelerator="auto", devices=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)
    dm.setup("test")

    model = LitCoMER.load_from_checkpoint(ckp)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    typer.run(main)