# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information with uncertainty hyperparmeter tuning."""
from __future__ import annotations

# Standard Library
import argparse
import logging
import os
import sys

# Scientific Libraries
import optuna
from optuna.samplers import TPESampler
from optuna_integration import PyTorchLightningPruningCallback

# Image Libraries
import cv2

# PyTorch
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins import _PLUGIN_INPUT, AsyncCheckpointIO

# First party imports
from models.attention.urr import URRResidualAttentionLightningModule
from models.attention.urr.utils import UncertaintyMode, URRSource
from models.attention.utils import REDUCE_TYPES
from scripts.attention_unet import ResidualTwoPlusOneDataModule
from utils import utils
from utils.global_progress_bar import BetterProgressBar
from utils.logging import LOGGING_FORMAT
from utils.types import (
    ClassificationMode,
    LoadingMode,
    MetricMode,
    ModelType,
    ResidualMode,
)

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))
IS_CHECKPOINTING = bool(os.environ.get("IS_CHECKPOINTING", "True"))
NUM_STEPS = -1 if IS_CHECKPOINTING else 1  # For testing whether this runs at all.
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 60))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
logger = logging.getLogger(__name__)


def objective(trial: optuna.trial.Trial) -> float:
    """Tune hyperparameters.

    Specifically, we tune for:
    - loss: {cross entropy, focal, weighted dice}
    - model type: { U-Net, UNet++ }
    - encoder: { se-resnet50, resnet50, tscSE-resnet50 }
    - attention reduction: { +, *, concat, weighted sum, weighted learnable }
    - single attention instance { True, False }
    - alpha: [0, 1]
    - beta: 1 - alpha
    - residual mode: { SNF, OF(CPU) }
    - histogram_equalisation: { True, False }
    - classification_mode { multiclass, multiclass(1, 2) }
    Honestly setting the config file options and handling them here is not very fun.
    Waiting for Python 3.14 for better support on running python processes from within
    python and being able to access its values.

    trial: Optuna trial object
    """
    # Model hyperparameters
    loss = trial.suggest_categorical(
        "loss", ["cross_entropy", "focal", "weighted_dice", None]
    )
    model_type = trial.suggest_categorical("model_type", ["UNET", "UNET_PLUS_PLUS"])
    encoder_name = trial.suggest_categorical(
        "encoder_name", ["se_resnet50", "resnet50", "tscse_resnet50"]
    )
    attention_reduction: REDUCE_TYPES = trial.suggest_categorical(
        "attention_reduction", ["sum", "prod", "cat", "weighted", "weighted_learnable"]
    )  # pyright: ignore[reportAssignmentType]
    single_attention_instance = trial.suggest_categorical(
        "single_attention_instance", [True, False]
    )
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = 1.0 - alpha
    residual_mode = trial.suggest_categorical(
        "residual_mode", ["SUBTRACT_NEXT_FRAME", "OPTICAL_FLOW_CPU"]
    )
    simple_residual_mode = {
        "SUBTRACT_NEXT_FRAME": "snf",
        "OPTICAL_FLOW_CPU": "of",
    }
    histogram_equalisation = trial.suggest_categorical(
        "histogram_equalisation", [True, False]
    )
    simple_classification_mode = {
        "MULTICLASS_MODE": "mc",
        "MULTICLASS_1_2_MODE": "mc12",
    }
    classification_mode = trial.suggest_categorical(
        "classification_mode",
        ["MULTICLASS_MODE", "MULTICLASS_1_2_MODE"],
    )
    learning_rate = trial.suggest_float("lr", low=1e-5, high=1e-3, log=True)

    version = "urr_{encoder_name}_{classification_mode}_{loss}_g_{attention_reduction}_{residual_mode}_alpha{alpha:.2f}_beta{beta:.2f}"

    version = version.format(
        encoder_name=encoder_name,
        classification_mode=simple_classification_mode[classification_mode],
        loss="dice" if loss is None else loss,
        attention_reduction=attention_reduction,
        residual_mode=simple_residual_mode[residual_mode],
        alpha=alpha,
        beta=beta,
    )

    model_type = ModelType[model_type]
    residual_mode = ResidualMode[residual_mode]
    classification_mode = ClassificationMode[classification_mode]

    model = URRResidualAttentionLightningModule(
        BATCH_SIZE,
        loss=loss,  # pyright: ignore[reportArgumentType]
        model_type=model_type,
        encoder_name=encoder_name,
        in_channels=1,
        classes=classification_mode.num_classes(),
        num_frames=10,
        optimizer="adamw",
        scheduler="cosine_anneal",
        learning_rate=learning_rate,
        attention_reduction=attention_reduction,
        total_epochs=NUM_EPOCHS,
        alpha=alpha,
        beta=beta,
        dl_classification_mode=classification_mode,
        eval_classification_mode=classification_mode,
        urr_source=URRSource.O3,
        uncertainty_mode=UncertaintyMode.URR,
        metric_mode=MetricMode.INCLUDE_EMPTY_CLASS,
        single_attention_instance=single_attention_instance,
        loading_mode=LoadingMode.GREYSCALE,
        residual_mode=residual_mode,
    )

    # Dataset hyperparameters
    datamodule = ResidualTwoPlusOneDataModule(
        data_dir="data/train_val",
        test_dir="data/test",
        indices_dir="data/indices",
        batch_size=BATCH_SIZE,
        frames=10,
        select_frame_method="specific",
        residual_mode=residual_mode,
        num_workers=8,
        loading_mode=LoadingMode.GREYSCALE,
        combine_train_val=True,
        histogram_equalize=histogram_equalisation,
        classification_mode=classification_mode,
    )

    save_dir = "./checkpoints/infarct/urr-residual-attention/lightning_logs"
    callbacks = [
        DeviceStatsMonitor(None),
        LearningRateMonitor("epoch", False, False),
        BetterProgressBar(),
        PyTorchLightningPruningCallback(trial, monitor="loss/val"),
    ]

    tensorboard_logger = None
    plugins: list[_PLUGIN_INPUT] | None = None
    if IS_CHECKPOINTING:
        callbacks += [
            ModelCheckpoint(  # Last epoch checkpoint.
                filename=utils.get_last_checkpoint_filename(version),
                save_last=True,
                save_weights_only=True,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            ),
            ModelCheckpoint(  # Best loss/val checkpoint
                filename=utils.get_checkpoint_filename(version),
                monitor="loss/val",
                save_last=False,
                save_weights_only=True,
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
        ]
        plugins = [AsyncCheckpointIO(None)]
        tensorboard_logger = TensorBoardLogger(save_dir=save_dir, version=version)

    trainer = L.Trainer(
        logger=tensorboard_logger,
        precision="bf16-mixed",
        max_epochs=NUM_EPOCHS,
        max_steps=NUM_STEPS,
        plugins=plugins,
        callbacks=callbacks,
        devices=1,
        accelerator="auto",
        strategy="auto",
        num_nodes=1,
        accumulate_grad_batches=utils.get_accumulate_grad_batches(1, model.batch_size),
    )

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["loss/val"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URR Residual Attention (Optuna)")
    _ = parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    sampler = TPESampler(multivariate=True, group=True)
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Persistence
        sampler=sampler,
        direction="minimize",
        pruner=pruner,
        study_name="URR Residual U-Net hyperparameters",
        load_if_exists=True,
    )

    # Logging
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    if os.getenv("DOCKERIZED", "0") != "0":
        # OPTIM: Prevent saving logs to file if in a docker container.
        file_handler = logging.FileHandler(
            filename="logs/optuna_urr_attention_unet.log"
        )
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [stdout_handler]

    logging.basicConfig(level=15, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)
    optuna.logging.get_logger("optuna").addHandler(stdout_handler)

    # WARNING: This should continue to run even after encountering most errors, which
    # may not be the wisest decision but it can also function as a test suite for
    # various combinations of parameters that I have yet to try.
    study.optimize(
        objective,
        n_trials=NUM_TRIALS,
        timeout=NUM_TRIALS * 120 * 60,  # num_trials * 120 min * 60s/min
        gc_after_trial=True,
        catch=[
            ValueError,
            AssertionError,
            SyntaxError,
            NotImplementedError,
            RuntimeError,
            cv2.error,
            torch.cuda.CudaError,
        ],
    )

    logger.info("Best value: %s, (params: %s)", study.best_value, study.best_params)
