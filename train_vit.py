import argparse
import torch
import einops

from vit_pytorch import vit
from data import FaceDataset
from loss import TripletLoss, TripletCosLoss


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Face ViT")
    
    parser.add_argument("--data_path", type=str, help="dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epoches", type=int, default=100, help="epoches")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--embd_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    
    args = parser.parse_args()
    return args


def train_net(args):
    torch.manual_seed(3948)
    
    model = vit.ViT(
        image_size=(112, 112),
        patch_size=(args.patch_size, args.patch_size),
        num_classes=args.embd_dim,
        dim=1024,
        depth=16,
        heads=8,
        mlp_dim=863,
    )
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    dataset = FaceDataset(args.data_path, train_ratio=1.0, input_shape=(3, 112, 112))
    dataset.train()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    criterion = TripletLoss(margin=0.1)
    # criterion = TripletCosLoss(margin=0.0)
    
    for epoch in range(args.epoches):
        train(model, dataloader, criterion, optim, epoch, device)
        torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch={epoch}.pth')
    
    
def train(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Forward
        embds = model(einops.rearrange(batch, 'b a c h w -> (b a) c h w'))
        embds = einops.rearrange(embds, '(b a) d -> b a d', a=3)
        batch_anchor_emb, batch_positive_emb, batch_negative_emb = torch.chunk(embds, 3, dim=1)
        
        # Loss
        loss = criterion(batch_anchor_emb, batch_positive_emb, batch_negative_emb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (i + 1) % 50 == 0:
            print(f'Epoch[{epoch+1}][{i+1}/{len(dataloader)}], loss={loss.item()}')
            
if __name__ == '__main__':
    args = parse_args()
    train_net(args)