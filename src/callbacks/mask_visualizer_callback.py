from typing import Dict, List

import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid

from src.transforms import inv_trans


class MaskVisualizer(Callback):
    def __init__(self):
        super().__init__()
        # self.threshold: float = threshold
        # self.rgb_color: Tuple[int, int, int] = rgb_color
        self.masks: List[Tensor] = []
        self.images: List[Tensor] = []

    def on_test_batch_end(  # noqa: WPS211
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: List[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.masks.append(outputs)
        self.images.append(batch[0])

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        masks = torch.cat(self.masks, dim=0)
        images = torch.cat(self.images, dim=0)
        images = [inv_trans(img) for img in images]

        # The below code doesn't for this dataset,
        # since draw_segmentation_masks requires image to be in rage 0 and 255

        # images_with_masks = [
        #     draw_segmentation_masks(
        #         img.to(torch.uint8),
        #         mask > self.threshold,
        #         colors=self.rgb_color,
        #     )
        #     for img, mask in zip(images, masks)]

        dict_of_grids = {
            'masks': masks,
            'images': images,
            # 'images with masks': images_with_masks
        }
        self.log_image(trainer, dict_of_grids)

    @staticmethod
    def log_image(trainer: Trainer, grid_dict: Dict[str, Tensor]) -> None:
        for name, grid in grid_dict.items():
            trainer.logger.experiment.add_image(
                name,
                img_tensor=make_grid(grid),
                global_step=trainer.current_epoch,
            )
