import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import Net
from utils import train, test

import wandb

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_layers', type=int, default = 3)
    parser.add_argument('--dropout', type = float, default = 0.1)


    parser.add_argument('--wb', type = bool, default = True)

    args = parser.parse_args()
    
    ###### Start W&B run and save hyperparameters ######
    if args.wb:
        wandb.init(project = "mnist_wandb_tutorial")
        wandb.config.update(args)
    ###########################
    device = torch.device("cuda:8")

    torch.manual_seed(args.seed)


    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': False,
        'shuffle': True
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.num_layers == 1:
        layers_dims = [784, 10]
    elif args.num_layers == 2:
        layers_dims = [784, 250, 10]
    elif args.num_layers == 3:
        layers_dims = [784, 250, 100, 10]

    model = Net(layers_dims, args.dropout).to(device)

    ######  Track weights and gradients ######
    if args.wb:
        wandb.watch(model, log = 'all', log_freq = 10 )
    ###########################

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_cum_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_cum_loss, test_acc = test(args, model, device, test_loader)

        print(f"Epoch: {epoch}/{args.epochs + 1} || Training Loss: {train_cum_loss} || Testing Loss: {test_cum_loss} || Testing accuracy: {test_acc}")

        if args.wb:
            ###### Track metrics ######
            wandb.log(
                {
                    'batch/epoch' : epoch,
                    'batch/train_loss': train_cum_loss,
                    'batch/test_loss': test_cum_loss,
                    'batch/test_accuracy':  test_acc,
                }
            )
            ###########################

        