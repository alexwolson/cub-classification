import argparse
from pathlib import Path
import pytorch_lightning as pl
import pytorch_lightning.loggers as WandbLogger
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--train-classification", type=bool, default=True)
    parser.add_argument("--train-regression", type=bool, default=True)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    wandb_logger = WandbLogger(
        project="CUB-Regression-Classification",
        name=f'{args.classification_weight}-{args.regression_weight}-{args.lr}',
        log_model=True,
        save_dir='reports'
    )

    wandb_logger.experiment.config.update(
        {
            'classification_weight':args.classification_weight,
            'regression_weight':args.regression_weight,
            'learning_rate':args.lr
        }
    )

    data_module = CUBDataModule(
        data_dir=Path(args.data_dir),
        batch_size=4
    )

    data_module.setup()

    model = CUBModel(
        num_classes=200,
        train_classification=args.train_classification,
        train_regression=args.train_regression,
        classification_weight=args.classification_weight,
        regression_weight=args.regression_weight
    )

    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger)

    trainer.fit(model, datamodule=data_module)