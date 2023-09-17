"""
Instruction Image Model with organ presence
Image Only: Yes (But with Yes/No Label for Liver/Bowel/Kidney... Presences)
Dimension: 2D
Backbone: Sam Base (Pretrained - Unfreeze - fine tune only head with projection layer) (256 output)
Dataset: SegmentationDatasetV2
"""
from typing import Any, Optional
import lightning as L

from timm.models import create_model

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy


import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from timm.models import create_model


class CNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(CNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.proj = nn.Conv2d(1, 3, 3)

        resnet = create_model("resnet152", pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            x = self.proj(x_3d[:, t, :, :, :])
            with torch.no_grad():
                x = self.resnet(x)  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv
            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, CNN_embed_dim)

        return cnn_embed_seq


class RNNDecoder(nn.Module):
    def __init__(
        self,
        CNN_embed_dim=300,
        h_RNN_layers=3,
        h_RNN=256,
        h_FC_dim=128,
        drop_p=0.3,
        num_classes=50,
    ):
        super(RNNDecoder, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 3, kernel_size=3)
        self.encoder = CNNEncoder()  # output -> (batch, time_step, 300)
        self.decoder = RNNDecoder(num_classes=100)  # output -> ()

        # TODO: head to optimise (concat in one linear)
        self.proj = nn.Linear(256, 100)
        self.head = nn.ModuleDict(
            {
                "bowel": nn.Sequential(nn.Linear(100, 1), nn.Sigmoid()),
                "extravasation": nn.Sequential(nn.Linear(100, 1), nn.Sigmoid()),
                "right_kidney": nn.Linear(100, 3),
                "left_kidney": nn.Linear(100, 3),
                "liver": nn.Linear(100, 3),
                "spleen": nn.Linear(100, 3),
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

        x = self.encoder(image)
        x = self.decoder(x)

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
