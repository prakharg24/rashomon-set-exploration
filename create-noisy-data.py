import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std, self.mean = std, mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_mnist_data_numpy(dataloader, mixup_transform=None):
    data_all, target_all = [], []
    for batch_idx, (data, target) in enumerate(dataloader):
        if mixup_transform is not None:
            data, target = mixup_transform(data, target)
            target = target.argmax(dim=1)

        data_all.append(data.detach().cpu().numpy())
        target_all.append(target.detach().cpu().numpy())

    return np.concatenate(data_all, axis=0), np.concatenate(target_all, axis=0)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset')
    parser.add_argument('--noisetype', type=str, default='gaussian',
                        help='dataset noise type')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mixup_transform = None
    assert args.dataset=='mnist'

    if args.noisetype=='none':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif args.noisetype=='gaussian':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            GaussianNoise(0., 1.)
        ])
    elif args.noisetype=='gaussian-high':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            GaussianNoise(0., 2.)
        ])
    elif args.noisetype=='mixup':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mixup_transform = v2.MixUp(num_classes=10)
    elif args.noisetype=='rotate':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(30)
        ])
    elif args.noisetype=='flip':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomHorizontalFlip()
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=train_transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1000)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)

    train_data, train_target = get_mnist_data_numpy(train_loader, mixup_transform=mixup_transform)
    test_data, test_target = get_mnist_data_numpy(test_loader, mixup_transform=mixup_transform)

    filename = '/home/mila/p/prakhar.ganesh/scratch/mnist-noise/' + args.noisetype + '_' + str(args.seed) + '.npy'
    with open(filename, 'wb') as f:
        np.save(f, train_data)
        np.save(f, train_target)
        np.save(f, test_data)
        np.save(f, test_target)

if __name__ == '__main__':
    main()