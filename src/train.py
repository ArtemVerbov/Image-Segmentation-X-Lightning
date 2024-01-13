from dataclasses import asdict
from typing import TYPE_CHECKING

import hydra
import lightning
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callbacks.debug import VisualizeBatch
from src.callbacks.mask_visualizer_callback import MaskVisualizer
from src.config import ExperimentConfig
from src.constants import CONFIG_PATH
from src.datamodule import SegmentationDataModule
from src.lightning_module import SegmentationLightningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from segmentation_models_pytorch import Unet
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


# noinspection PyDataclass
@hydra.main(config_path=str(CONFIG_PATH), config_name='conf', version_base='1.2')
def train(cfg: 'DictConfig'):  # noqa: WPS210

    experiment_config: ExperimentConfig = hydra.utils.instantiate(cfg.experiment_config)
    opt: 'Optimizer' = hydra.utils.instantiate(cfg.optimizer)
    scheduler: 'LRScheduler' = hydra.utils.instantiate(cfg.scheduler)
    segmentation_model: 'Unet' = hydra.utils.instantiate(cfg.model)

    lightning.seed_everything(0)
    datamodule = SegmentationDataModule(cfg=experiment_config.data_config)

    if experiment_config.project_config.track_in_clearml:
        Task.force_requirements_env_freeze()
        task = Task.init(
            project_name=experiment_config.project_config.project_name,
            task_name=experiment_config.project_config.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
        )
        # Stores yaml config as a dictionary in clearml
        task.connect(asdict(experiment_config))
        task.connect_configuration(datamodule.transforms.get_train_transforms(), name='transformations')

    model = SegmentationLightningModule(
        model=segmentation_model,
        module_cfg=experiment_config.lightning_module_config,
        optimizer=opt,
        scheduler=scheduler,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    visualize = VisualizeBatch(every_n_epochs=5)
    masks_visualizer = MaskVisualizer()
    early_stopping = EarlyStopping(monitor='mean_valid_loss', patience=5)
    check_points = ModelCheckpoint(monitor='mean_valid_loss', mode='min', verbose=True)

    trainer = Trainer(
        **asdict(experiment_config.trainer_config),
        callbacks=[
            lr_monitor,
            visualize,
            masks_visualizer,
            early_stopping,
            check_points,
        ],
        # overfit_batches=10
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path='best',
    )


if __name__ == '__main__':
    train()
