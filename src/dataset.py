import os
from pathlib import Path
from typing import Optional, Tuple

import albumentations as albu
import cv2
from numpy.typing import NDArray
from torch import Tensor, clip
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_paths: Path, transforms: Optional[albu.Compose] = None):
        self.imagePaths = image_paths
        self.transforms = transforms

        if self.transforms is None:
            raise ValueError('At least np.ndarray to torch.Tensor transformation should be defined.')

        self.subdir_names = os.listdir(self.imagePaths)

    def __len__(self) -> int:
        # return the number of total samples contained in the dataset
        return len(self.subdir_names)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        base_path = self.imagePaths / self.subdir_names[idx]
        image_path = str(base_path / 'images' / f'{self.subdir_names[idx]}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            mask = self.get_mask(base_path)
            transformed = self.transforms(image=image, mask=mask)
            mask = clip(transformed['mask'], 0, 1).float().unsqueeze(0)
        except FileNotFoundError:
            transformed = self.transforms(image=image)
            mask = 0

        return transformed['image'].float(), mask

    # noinspection PyTypeChecker
    @staticmethod
    def get_mask(path) -> NDArray:
        path = path / 'masks'
        masks_list = []
        for mask_part in os.listdir(path):
            mask = cv2.imread(str(path / mask_part), cv2.IMREAD_GRAYSCALE)
            masks_list.append(mask)
        return sum(masks_list)
