import os
from typing import Tuple
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
import cv2
from numpy.typing import NDArray
from torchvision.datasets import ImageFolder, DatasetFolder
from transforms import Transforms


class SegmentationDataset(Dataset):

    def __init__(self, image_paths, transforms):
        self.imagePaths = image_paths
        self.transforms = transforms

        if self.transforms is None:
            raise ValueError('At least np.ndarray to torch.Tensor transformation should be defined.')

        self.subdir_names = os.listdir(self.imagePaths)

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.subdir_names)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        base_path = self.imagePaths / self.subdir_names[idx]
        image_path = str(base_path / 'images' / f'{self.subdir_names[idx]}.png')

        mask = self.get_mask(base_path)/255.0
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image, mask=mask)

        return transformed['image'].float(), transformed['mask'].float().unsqueeze(0)

    @staticmethod
    def get_mask(path):
        path = path / 'masks'
        masks_list = list()
        for mask_part in os.listdir(path):
            mask = cv2.imread(str(path / mask_part), cv2.IMREAD_GRAYSCALE)
            masks_list.append(mask)
        return sum(masks_list)


if __name__ == '__main__':
    from pathlib import Path
    from typing import Dict, Optional

    import torch
    from clearml import Dataset as ClearmlDataset

    data_path = Path(ClearmlDataset.get(dataset_name='image_segmentation_dataset').get_local_copy())
    trans = Transforms(64, 64).compose('fit')
    data = SegmentationDataset(data_path / 'train', trans)

    for i in iter(data):
        print(i)
        break
    # print(next(iter(data)))

