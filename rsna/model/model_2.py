"""
Instruction Image Model with organ presence
Backbone: ResnetV2 (100 output)
Dataset: SegmentationDatasetV2
"""
from typing import Any, Optional
import lightning as L

from timm.models import create_model

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 3, kernel_size=3)
        self.backbone = create_model(
            "resnetv2_50x3_bit", pretrained=True, num_classes=100
        )  # output: (B, 1000)
        # TODO: head to optimise (concat in one linear)
        self.head = nn.ModuleDict(
            {
                "bowel": nn.Sequential(nn.Linear(13, 1), nn.Sigmoid()),
                "extravasation": nn.Sequential(nn.Linear(13, 1), nn.Sigmoid()),
                "right_kidney": nn.Linear(13, 3),
                "left_kidney": nn.Linear(13, 3),
                "liver": nn.Linear(13, 3),
                "spleen": nn.Linear(13, 3),
            }
        )

        # fmt:off
        self.bowel_train_accuracy             = Accuracy(task="binary", num_classes=2)
        self.extravasation_train_accuracy     = Accuracy(task="binary", num_classes=2)
        self.kidney_train_accuracy            = Accuracy(task="multiclass", num_classes=3)
        self.liver_train_accuracy             = Accuracy(task="multiclass", num_classes=3)
        self.spleen_train_accuracy            = Accuracy(task="multiclass", num_classes=3)
        
        self.bowel_val_accuracy             = Accuracy(task="binary", num_classes=2)
        self.extravasation_val_accuracy     = Accuracy(task="binary", num_classes=2)
        self.kidney_val_accuracy            = Accuracy(task="multiclass", num_classes=3)
        self.liver_val_accuracy             = Accuracy(task="multiclass", num_classes=3)
        self.spleen_val_accuracy            = Accuracy(task="multiclass", num_classes=3)
        # fmt:on

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def step(self, batch, batch_idx, train=True):
        (
            image,
            mask,
            # 1 if liver is in the image else 0, same goes to others
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
        ) = batch

        x = self.backbone(self.conv2d(image))

        # fmt: off
        right_kidney        = right_kidney.unsqueeze(-1)
        left_kidney         = left_kidney.unsqueeze(-1)
        liver               = liver.unsqueeze(-1)
        spleen              = spleen.unsqueeze(-1)

        # disable learning if no organ found in the image
        y_bowel             = bowel * self.head["bowel"](x).view(-1)
        y_extravasation     = self.head["extravasation"](x).view(-1)
        y_kidney            = right_kidney * self.head["right_kidney"](x) + left_kidney * self.head["left_kidney"](x)
        y_liver             = liver * self.head["liver"](x)
        y_spleen            = spleen * self.head["spleen"](x)
        
        groundtruth_bowel   = bowel * bowel_healthy
        groundtruth_kidney  = right_kidney * left_kidney * torch.stack([kidney_healthy, kidney_low, kidney_high], dim=-1)
        groundtruth_liver   = liver * torch.stack([liver_healthy, liver_low, liver_high], dim=-1)
        groundtruth_spleen  = spleen * torch.stack([spleen_healthy, spleen_low, spleen_high], dim=-1)
        
        # Compute loss
        loss_bowel          = F.binary_cross_entropy(y_bowel, groundtruth_bowel)
        loss_extravasation  = F.binary_cross_entropy(y_extravasation, extravasation_healthy)
        loss_kidney         = F.cross_entropy(y_kidney, groundtruth_kidney)
        loss_liver          = F.cross_entropy(y_liver, groundtruth_liver)
        loss_spleen         = F.cross_entropy(y_spleen, groundtruth_spleen)

        # Element sum
        loss_total = (
            loss_bowel + loss_extravasation + loss_kidney + loss_liver + loss_spleen
        )
        
        if train:
            # Compute accuracy
            self.bowel_train_accuracy(y_bowel, groundtruth_bowel)
            self.extravasation_train_accuracy(y_extravasation, extravasation_healthy)
            self.kidney_train_accuracy(y_kidney, groundtruth_kidney)
            self.liver_train_accuracy(y_liver, groundtruth_liver)
            self.spleen_train_accuracy(y_spleen, groundtruth_spleen)
            # fmt: on

            # logging
            self.log("train_loss_bowel", loss_bowel)
            self.log("train_loss_extravasation", loss_extravasation)
            self.log("train_loss_kidney", loss_kidney)
            self.log("train_loss_liver", loss_liver)
            self.log("train_loss_spleen", loss_spleen)
            self.log("train_loss_total", loss_total)

            self.log("train_acc_bowel", self.bowel_train_accuracy)
            self.log("train_acc_extravasation", self.extravasation_train_accuracy)
            self.log("train_acc_kidney", self.kidney_train_accuracy)
            self.log("train_acc_liver", self.liver_train_accuracy)
            self.log("train_acc_spleen", self.spleen_train_accuracy)
        else:
            # Compute accuracy
            self.bowel_val_accuracy(y_bowel, groundtruth_bowel)
            self.extravasation_val_accuracy(y_extravasation, extravasation_healthy)
            self.kidney_val_accuracy(y_kidney, groundtruth_kidney)
            self.liver_val_accuracy(y_liver, groundtruth_liver)
            self.spleen_val_accuracy(y_spleen, groundtruth_spleen)
            # fmt: on

            # logging
            self.log("val_loss_bowel", loss_bowel)
            self.log("val_loss_extravasation", loss_extravasation)
            self.log("val_loss_kidney", loss_kidney)
            self.log("val_loss_liver", loss_liver)
            self.log("val_loss_spleen", loss_spleen)
            self.log("val_loss_total", loss_total)

            self.log("val_acc_bowel", self.bowel_val_accuracy)
            self.log("val_acc_extravasation", self.extravasation_val_accuracy)
            self.log("val_acc_kidney", self.kidney_val_accuracy)
            self.log("val_acc_liver", self.liver_val_accuracy)
            self.log("val_acc_spleen", self.spleen_val_accuracy)
        return loss_total

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, train=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, train=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
