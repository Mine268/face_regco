import argparse
import torch
import einops
import random
from tqdm import tqdm

from vit_pytorch import vit
from data import FaceDataset
from loss import TripletLoss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_path = './test_split'

torch.manual_seed(3948)

model = vit.ViT(
    image_size=(112, 112),
    patch_size=(16, 16),
    num_classes=256,
    dim=400,
    depth=16,
    heads=8,
    mlp_dim=340,
)
model = model.to(device)
model.eval()

dataset = FaceDataset(train_path)

with torch.no_grad():
    for j in range(20):
        p = './checkpoints/checkpoint_epoch=' + str(j) + '.pth'
        w = torch.load(p)
        model.load_state_dict(w)
        
        def auc(model, dataset):
            n = 0
            for i in tqdm(range(len(dataset))):
                embeds = model(dataset[i].to(device))
                s1 = torch.nn.functional.cosine_similarity(embeds[0].unsqueeze(0), embeds[1].unsqueeze(0))
                s2 = torch.nn.functional.cosine_similarity(embeds[0].unsqueeze(0), embeds[2].unsqueeze(0))
                if s1 > s2:
                    n += 1
                elif torch.abs(s1 - s2) < 1e-5:
                    n += 0.5
            return n / len(dataset)
        print(auc(model, dataset))
        
    w = torch.load('./checkpoints/checkpoint_epoch=19.pth')
    model.load_state_dict(w)
    def roc(model, dataset, size=None):
        if size is None:
            size = len(dataset)
        
        size = min(size, len(dataset))
        
        # 初始化用于存储 TPR 和 FPR 的列表
        tpr_list = [0] * 101
        fpr_list = [0] * 101
        
        # 确定正样本和负样本的总数
        total_positives = size
        total_negatives = size

        for i in tqdm(range(size)):
            # 获取模型嵌入
            embeds = model(dataset[i].to(device))
            
            # 计算正样本和负样本的相似度
            s_pos = torch.nn.functional.cosine_similarity(embeds[0].unsqueeze(0), embeds[1].unsqueeze(0))
            s_neg = torch.nn.functional.cosine_similarity(embeds[0].unsqueeze(0), embeds[2].unsqueeze(0))
            
            # 将相似度映射到 [0, 100] 范围
            s_pos = (s_pos + 1) * 50
            s_neg = (s_neg + 1) * 50

            for t in range(101):
                # 确定当前阈值
                threshold = 100 - t

                # 更新 TPR 和 FPR
                if s_pos >= threshold:
                    tpr_list[t] += 1
                if s_neg >= threshold:
                    fpr_list[t] += 1

        # 将 TPR 和 FPR 转换为比率
        tpr_list = [x / total_positives for x in tpr_list]
        fpr_list = [x / total_negatives for x in fpr_list]

        return tpr_list, fpr_list
    print(roc(model, dataset))
