import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR
from scipy.stats import wasserstein_distance


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std, self.mean = std, mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_dataset(dataset_name):
    mixup_transform = None
    if dataset_name=='mnist':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name=='mnist-noise':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            GaussianNoise(0., 1.)
        ])
    elif dataset_name=='mnist-noise-high':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            GaussianNoise(0., 2.)
        ])
    elif dataset_name=='mnist-mixup':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mixup_transform = v2.MixUp(num_classes=10)
    elif dataset_name=='mnist-rotate':
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(30)
        ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=train_transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=test_transform)
    
    train_data, train_label = [], []
    for data, label in dataset1:
        train_data.append(data.detach().numpy())
        train_label.append(label)
    train_data, train_label = np.array(train_data), np.array(train_label)

    test_data, test_label = [], []
    for data, label in dataset2:
        test_data.append(data.detach().numpy())
        test_label.append(label)
    test_data, test_label = np.array(test_data), np.array(test_label)

    return train_data, train_label, test_data, test_label

def display_image(img, fname):
    plt.imsave(fname, img.reshape(28,28), cmap="gray")

def ecdf(data):
    values, bins = np.histogram(data, bins=np.arange(-1, 3, 0.1), density=True)

    return (bins, values)

def main():
    dataset_a_name = 'mnist-rotate'
    dataset_b_name = 'mnist-noise'

    torch.manual_seed(0)

    train_data_a, train_label_a, test_data_a, test_label_a = load_dataset(dataset_a_name)
    train_data_b, train_label_b, test_data_b, test_label_b = load_dataset(dataset_b_name)

    print("Data Loaded")

    wass_dist_mat = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            train_data_a_i = train_data_a[train_label_a==i].reshape(-1, 28*28)
            train_data_a_j = train_data_a[train_label_a==j].reshape(-1, 28*28)

            wass_dist_arr = []
            for ite in range(28*28):
                distrib_i_ite = ecdf(train_data_a_i[:, ite])
                distrib_j_ite = ecdf(train_data_a_j[:, ite])
                wass_dist_arr.append(wasserstein_distance(distrib_i_ite[1], distrib_j_ite[1]))

            wass_dist_mat[i, j] = np.mean(wass_dist_arr)
            wass_dist_mat[j, i] = wass_dist_mat[i, j]

    print(np.mean(wass_dist_mat, axis=1))

    # for ite in range(10):
    #     display_image(np.mean(train_data_a[train_label_a==ite], axis=0), 'images/average_train_%d.png' % ite)
    #     display_image(np.mean(train_data_b[train_label_b==ite], axis=0), 'images/average_train_noise_%d.png' % ite)
    
if __name__ == '__main__':
    main()