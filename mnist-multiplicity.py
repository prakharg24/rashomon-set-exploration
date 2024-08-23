import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std, self.mean = std, mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def pred(model, device, test_loader):
    model.eval()
    pred_arr, lbl_arr = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            pred_arr.extend(pred.detach().cpu().tolist())
            lbl_arr.extend(target.detach().cpu().tolist())

    return np.array(pred_arr), np.array(lbl_arr)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST Multiplicity')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to train (choose from: mnist|mnist-noise)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset2 = datasets.MNIST('../data', train=False,
                       transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    all_pred = []
    for seed in tqdm(range(100)):
        args.seed = seed
        filename = '/home/mila/p/prakhar.ganesh/scratch/rashomon-set-mnist/model_%s_%d.pt' % (args.dataset, args.seed)
        model.load_state_dict(torch.load(filename))

        pred_arr, lbl_arr = pred(model, device, test_loader)

        acc = np.mean(pred_arr==lbl_arr)
        if acc > 0.98:
            all_pred.append(pred_arr)
    
    print("Number of Models: %d" % len(all_pred))
    all_pred = np.array(all_pred)

    def get_ambiguity(pred_mat):
        amb_bool = pred_mat[0] == pred_mat
        amb = 1 - np.mean(np.all(amb_bool, axis=0))
        return amb

    print("Ambiguity for the Dataset: %.2f%%" % (get_ambiguity(all_pred)*100))
    amb_per_class = []
    for cls_ind in range(10):
        amb_per_class.append(get_ambiguity(all_pred[:, lbl_arr==cls_ind])*100)
        print("Ambiguity for Class %d: %.2f%%" % (cls_ind, get_ambiguity(all_pred[:, lbl_arr==cls_ind])*100))
    
    plt.bar(range(10), amb_per_class)
    plt.title('MNIST-Rotate')
    plt.xlabel('Classes')
    plt.ylabel('Ambiguity (Percentage)')
    plt.savefig('mnist-rotate_per_class_amb.png')
    
if __name__ == '__main__':
    main()