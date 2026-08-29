import torch
from torch import nn
import time
import matplotlib.pyplot as plt


def train(net:nn.Module, train_iter, val_iter, lr, wd, num_epochs, device):
    print(f'train on {device}')
    net = net.to(device)
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    train_ls, val_ls = [], []
    for epoch in range(num_epochs):
        l_sum, total = 0, 0
        net.train()
        start = time.time()
        for X, Y in train_iter: # [B, T]
            X, Y = X.to(device), Y.to(device)
            Y_hat = net(X) # [B, T, V]
            l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, 1))
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            l_sum += l.sum().item()
            total += Y.numel()
        train_ls.append(l_sum / total)
        if epoch == 0:
            speed =  total / (time.time() - start)
            print(f'{speed:.1f} token/s')

        l_val, n = 0, 0
        net.eval()
        for X, Y in val_iter:
            X, Y = X.to(device), Y.to(device)
            Y_hat = net(X) # [B, T, V]
            l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1, 1))
            l_val += l.sum().item()
            n += Y.numel()
        val_ls.append(l_val / n)

        if epoch % 10 == 9:
            print(f'epoch:{epoch+1}, train_l:{train_ls[-1]:.4f}, val_l:{val_ls[-1]:.4f}')

    print('finish training!')
    return train_ls, val_ls

def plot_loss(train_ls, val_ls, save=None):
    num_epochs = len(train_ls)
    x = list(range(num_epochs))
    plt.plot(x, train_ls, label='train loss')
    plt.plot(x, val_ls, label='val loss')
    plt.legend()
    if save is not None:
        plt.savefig(save)
    plt.show()
