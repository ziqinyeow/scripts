from typing import Literal, Optional

import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

import torchvision.transforms as T


def split(
    dataset, train_size: float = 0.8, generator: Optional[torch.Generator] = None
):
    if not generator:
        generator = torch.Generator().manual_seed(42)

    return random_split(dataset, [train_size, 1 - train_size], generator=generator)


class SegmentationDataset(data.Dataset):
    def __init__(self, transform=None, dimension: Literal["2d"] = "2d"):
        self.transform = transform if transform else T.Resize((512, 512))
        self.dimension = dimension
        self.dataset = load_dataset("ziq/RSNA-ATD2023", split="train").with_format(
            "torch", dtype=torch.float32
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (
            patient_id,
            series_id,
            frame_id,
            image,
            mask,
            liver,
            spleen,
            right_kidney,
            left_kidney,
            bowel,
            aortic_hu,
            incomplete_organ,
            bowel_healthy,
            bowel_injury,
            extravasation_healthy,
            extravasation_injury,
            kidney_healthy,
            kidney_low,
            kidney_high,
            liver_healthy,
            liver_low,
            liver_high,
            spleen_healthy,
            spleen_low,
            spleen_high,
            any_injury,
        ) = self.dataset[index].values()

        image = image.view(-1, image.shape[0], image.shape[1])
        image = self.transform(image)

        return (
            image,
            liver,
            spleen,
            right_kidney,
            left_kidney,
            bowel,
        ), (
            bowel_healthy,
            bowel_injury,
            extravasation_healthy,
            extravasation_injury,
            kidney_healthy,
            kidney_low,
            kidney_high,
            liver_healthy,
            liver_low,
            liver_high,
            spleen_healthy,
            spleen_low,
            spleen_high,
        )


class SegmentationDatasetV2(data.Dataset):
    def __init__(self, transform=None, dimension: Literal["2d"] = "2d"):
        self.transform = transform if transform else T.Resize((512, 512))
        self.dimension = dimension
        self.dataset = load_dataset("ziq/RSNA-ATD2023", split="train").with_format(
            "torch", dtype=torch.float32
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (
            patient_id,
            series_id,
            frame_id,
            image,
            mask,
            liver,
            spleen,
            right_kidney,
            left_kidney,
            bowel,
            aortic_hu,
            incomplete_organ,
            bowel_healthy,
            bowel_injury,
            extravasation_healthy,
            extravasation_injury,
            kidney_healthy,
            kidney_low,
            kidney_high,
            liver_healthy,
            liver_low,
            liver_high,
            spleen_healthy,
            spleen_low,
            spleen_high,
            any_injury,
        ) = self.dataset[index].values()

        image, mask = image.unsqueeze(0), mask.unsqueeze(0)
        image, mask = self.transform(image), self.transform(mask)
        image /= 255
        mask /= 5

        return (
            image,
            mask,
            liver,
            spleen,
            right_kidney,
            left_kidney,
            bowel,
        ), (
            bowel_healthy,
            bowel_injury,
            extravasation_healthy,
            extravasation_injury,
            kidney_healthy,
            kidney_low,
            kidney_high,
            liver_healthy,
            liver_low,
            liver_high,
            spleen_healthy,
            spleen_low,
            spleen_high,
        )
