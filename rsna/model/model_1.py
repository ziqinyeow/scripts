import lightning as L

# from lightning.pytorch.loggers.wandb import WandbLogger
from timm.models import create_model

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 3, kernel_size=3)
        self.backbone = create_model("fastvit_ma36")  # output: (B, 1000)
        # TODO: head to optimise (concat in one linear)
        self.head = nn.ModuleDict(
            {
                "bowel": nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid()),
                "extravasation": nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid()),
                "right_kidney": nn.Linear(1000, 3),
                "left_kidney": nn.Linear(1000, 3),
                "liver": nn.Linear(1000, 3),
                "spleen": nn.Linear(1000, 3),
            }
        )

        self.bowel_accuracy = Accuracy(task="binary", num_classes=2)
        self.extravasation_accuracy = Accuracy(task="binary", num_classes=2)
        self.kidney_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.liver_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.spleen_accuracy = Accuracy(task="multiclass", num_classes=3)

    def training_step(self, batch, batch_idx):
        (
            image,
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
        
        # Compute loss
        loss_bowel          = F.binary_cross_entropy(y_bowel, bowel * bowel_healthy)
        loss_extravasation  = F.binary_cross_entropy(y_extravasation, extravasation_healthy)
        loss_kidney         = F.cross_entropy(y_kidney, right_kidney * left_kidney * torch.stack([kidney_healthy, kidney_low, kidney_high], dim=-1))
        loss_liver          = F.cross_entropy(y_liver, liver * torch.stack([liver_healthy, liver_low, liver_high], dim=-1))
        loss_spleen         = F.cross_entropy(y_spleen, spleen * torch.stack([spleen_healthy, spleen_low, spleen_high], dim=-1))

        # Element sum
        loss_total = (
            loss_bowel + loss_extravasation + loss_kidney + loss_liver + loss_spleen
        )
        
        # Compute accuracy
        self.bowel_accuracy(y_bowel, bowel_healthy)
        self.extravasation_accuracy(y_extravasation, extravasation_healthy)
        self.kidney_accuracy(y_kidney, kidney_healthy)
        self.liver_accuracy(y_liver, liver_healthy)
        self.spleen_accuracy(y_spleen, spleen_healthy)
        # fmt: on

        # logging
        self.log("loss_bowel", loss_bowel)
        self.log("loss_extravasation", loss_extravasation)
        self.log("loss_kidney", loss_kidney)
        self.log("loss_liver", loss_liver)
        self.log("loss_spleen", loss_spleen)
        self.log("loss_total", loss_total)

        self.log("acc_bowel", self.bowel_accuracy)
        self.log("acc_extravasation", self.extravasation_accuracy)
        self.log("acc_kidney", self.kidney_accuracy)
        self.log("acc_liver", self.liver_accuracy)
        self.log("acc_spleen", self.spleen_accuracy)

        return loss_total

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
