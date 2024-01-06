from typing import TYPE_CHECKING, Dict, List, Optional

import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric

from src.metrics import get_metrics

if TYPE_CHECKING:
    from segmentation_models_pytorch import Unet
    from torch.optim import Adam

    from src.config import LightningModuleConfig


class SegmentationLightningModule(LightningModule):  # noqa: WPS214
    def __init__(
        self,
        model: 'Unet',
        module_cfg: 'LightningModuleConfig',
        # class_to_idx: Dict[str, int],
        optimizer: 'Adam',
        scheduler: Optional['ReduceLROnPlateau'] = None,
    ):
        super().__init__()
        metrics = get_metrics()

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.module_cfg = module_cfg
        self.model = model()

        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]):  # noqa: WPS210
        images, targets = batch
        logits = self.forward(images)
        loss = func.binary_cross_entropy(logits, targets)
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)

        return {'loss': loss, 'preds': logits, 'target': targets}

    def validation_step(self, batch: List[Tensor], batch_index: int):  # noqa: WPS210
        images, targets = batch
        logits = self.forward(images)
        loss = func.binary_cross_entropy(logits, targets)
        self._valid_loss(loss)

        self._valid_metrics(logits, targets.long())
        self.log('val_loss', loss, on_step=False, prog_bar=False, logger=True)

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, targets = batch
        logits = self(images)

        return logits

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss, on_step=False, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log('mean_valid_loss', self._valid_loss, on_step=False, prog_bar=True, on_epoch=True)
        self.log_dict(self._valid_metrics, prog_bar=True, on_epoch=True)

    # noinspection PyCallingNonCallable
    def configure_optimizers(self) -> Dict:
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.module_cfg.interval,
                    'frequency': self.module_cfg.optimizer_frequency,
                    'monitor': self.module_cfg.monitor,
                },
            }
        return {'optimizer': optimizer}
