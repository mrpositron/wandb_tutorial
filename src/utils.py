import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    cum_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


        if args.wb:
            wandb.log(
                {
                    'iter/train_loss': loss.item(), 
                }
            )

        cum_loss += loss.item()
    return cum_loss/len(train_loader)

@torch.no_grad()
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    correct = 100. * correct / len(test_loader.dataset)


    return test_loss, correct