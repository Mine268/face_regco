import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(dim=1, keepdim=True)
        distance_negative = (anchor - negative).pow(2).sum(dim=1, keepdim=True)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
