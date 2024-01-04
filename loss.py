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


class TripletCosLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletCosLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        distance_positive = torch.sum(anchor * positive, dim=1)
        distance_negative = torch.sum(anchor * negative, dim=1)
        losses = 0.1 * torch.sigmoid(distance_positive + self.margin) - torch.sigmoid(distance_negative)
        return losses.mean()

