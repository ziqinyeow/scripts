from typing import Literal, Optional

import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

from PIL import Image
import numpy as np
import pandas as pd

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
        self.transform = (
            transform if transform else T.Resize((512, 512), antialias=True)
        )
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


class VideoSegmentationDataset(data.IterableDataset):
    def __init__(
        self,
        d=16,
        size=512,
        train=True,
        transform=None,
        images="images/",
        masks="masks/",
        metadata="metadata.csv",
    ):
        self.d = d
        self.size = size
        self.train = train
        self.transform = (
            transform if transform else T.Resize((self.size, self.size), antialias=True)
        )
        self.metadata = pd.read_csv(metadata)
        self.images, self.masks = images, masks

    def pad_and_chunk_series(self, series, d):
        l = len(series)
        max = l + d - l % d
        chunks = [series[i : i + d] for i in range(0, max, d)]
        last = len(chunks[-1])
        if last != d:
            ls = chunks[0][0].tolist()
            c = [-1] + ls[1:3] + [-1] + [0] * 5 + ls[9:]
            chunks[-1] = np.append(chunks[-1], [c] * (d - last), axis=0)
        chunks = np.array(chunks)
        return chunks

    def generator(self):
        metadata, d = self.metadata, self.d
        series = metadata["series_id"].unique()

        for s in series:
            ndseries = metadata[metadata["series_id"] == s].to_numpy()
            labels, padded_ndseries = torch.from_numpy(
                ndseries[0, 3:].astype(np.float32)
            ), self.pad_and_chunk_series(ndseries, d)
            shape = np.array(
                Image.open(f"{self.images}/{padded_ndseries[0][0][0]}")
            ).shape
            for i, batch in enumerate(padded_ndseries):
                if not self.train and i != len(padded_ndseries) - 1:
                    continue
                if self.train and i == len(padded_ndseries) - 1:
                    continue
                paths = batch[:, 0]
                im = np.array(
                    [
                        np.array(Image.open(f"{self.images}/{p}"))
                        if p != -1
                        else np.zeros(shape)
                        for p in paths
                    ]
                )
                im = np.expand_dims(im, 1).astype(np.float32)
                im = torch.from_numpy(im)
                im = self.transform(im) / 255

                masks = np.array(
                    [
                        np.array(Image.open(f"{self.masks}/{p}"))
                        if p != -1
                        else np.zeros(shape)
                        for p in paths
                    ]
                )
                masks = np.expand_dims(masks, 1).astype(np.float32)
                masks = torch.from_numpy(masks)
                masks = self.transform(im) / 5

                yield (
                    im,
                    masks,
                    *torch.tensor_split(labels[0:5], 5),
                ), torch.tensor_split(labels[5:], 13)

    def __iter__(self):
        return self.generator()
