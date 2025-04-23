import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    classification_weight = trial.suggest_float("classification_weight", 0.1, 1.)
    regression_weight = trial.suggest_float("regression_weight", 0.1, 1.)
    
    wandb_logger = WandbLogger(
        project="CUB-Hypertuning",
        log_model=True,
        name=f"trial-{trial.number}"
    )

    wandb_logger.experiment.config.update({
        "lr":lr,
        "classification_weight":classification_weight,
        "regression_weight":regression_weight
    })

    data_module = CUBDataModule(
        data_dir=Path(args.data_dir),
        batch_size=4
    )

    data_module.setup()

    model = CUBModel(
        num_classes=200,
        train_classification=True,
        train_regression=True,
        classification_weight=classification_weight,
        regression_weight=regression_weight,
        lr=lr
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_combined_metric',
        patience=2,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, callbacks=[early_stopping_callback])

    trainer.fit(model, datamodule=data_module)

    # loss = trainer.evaluate(model, )

    wandb_logger.experiment.finish()

    return trainer.callback_metrics["val_combined_metric"].item()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--n_trials", type=int, default=20)
    # parser.add_argument("--train-classification", type=bool, default=True)
    # parser.add_argument("--train-regression", type=bool, default=True)
    # parser.add_argument("--classification-weight", type=float, default=1.0)
    # parser.add_argument("--regression-weight", type=float, default=1.0)
    # parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    study = optuna.create_study(
        direction='maximize',
        study_name='CUB-Class-and-Regr',
        storage="sqlite:///cub_optuna_study.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=args.n_trials)

    # wandb_logger = WandbLogger(
    #     project="CUB-Regression-Classification",
    #     name=f'{args.classification_weight}-{args.regression_weight}-{args.lr}',
    #     log_model=True,
    #     save_dir='reports'
    # )

    # wandb_logger.experiment.config.update(
    #     {
    #         'classification_weight':args.classification_weight,
    #         'regression_weight':args.regression_weight,
    #         'learning_rate':args.lr
    #     }
    # )

    # data_module = CUBDataModule(
    #     data_dir=Path(args.data_dir),
    #     batch_size=4
    # )

    # data_module.setup()

    # model = CUBModel(
    #     num_classes=200,
    #     train_classification=args.train_classification,
    #     train_regression=args.train_regression,
    #     classification_weight=args.classification_weight,
    #     regression_weight=args.regression_weight
    # )

    # trainer = pl.Trainer(max_epochs=5, logger=wandb_logger)

    # trainer.fit(model, datamodule=data_module)