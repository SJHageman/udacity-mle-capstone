import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage import filters, transform
from skimage.io import imread
from skimage import img_as_ubyte
from typing import Tuple

import pandas as pd

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.cosine_similarity(F.normalize(output1), F.normalize(output2))
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def preprocess_image(img,final_img_size,img_size =(170, 242),input_size =(150, 220)):
    img = img.astype(np.uint8)
    resized = resize_image(255 - normalize_image(img, final_img_size), img_size)
    if input_size is not None and input_size != img_size:
        cropped = crop_center(resized, input_size)
    else:
        cropped = resized
    return cropped

def normalize_image(img,final_img_size = (1000, 1400)):
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    threshold = filters.threshold_otsu(img)

    binarized_image = blurred_image > threshold
    rows, columns = np.where(binarized_image == 0)
    row_com = int(rows.mean() - rows.min())
    column_com = int(columns.mean() - columns.min())

    cropped = img[rows.min(): rows.max(), columns.min(): columns.max()]

    img_rows, img_cols = cropped.shape
    max_rows, max_cols = final_img_size

    row_start = max_rows // 2 - row_com
    column_start = max_cols // 2 - column_com

    if img_rows > max_rows:
        row_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_rows, :]
        img_rows = max_rows
    else:
        extra_rows = (row_start + img_rows) - max_rows
        if extra_rows > 0:
            row_start = row_start - extra_rows
        if row_start < 0:
            row_start = 0

    if img_cols > max_cols:
        column_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_cols]
        img_cols = max_cols
    else:
        extra_columns = (column_start + img_cols) - max_cols
        if extra_columns > 0:
            column_start = column_start - extra_columns
        if column_start < 0:
            column_start = 0
    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    normalized_image[row_start:row_start + img_rows, column_start:column_start + img_cols] = cropped
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def resize_image(img,size):
    height, width = size
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)
    
    if width_ratio > height_ratio:
        start = int(round((resize_width-width)/2.0))
        return img[:, start:start + width]
    else:
        start = int(round((resize_height-height)/2.0))
        return img[start:start + height, :]


def crop_center(img, size):
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped

def load_image(path):
    return img_as_ubyte(imread(path, as_gray=True))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        input1 = input1.view(-1, 1, 150, 220).float().div(255)
        input2 = input2.view(-1, 1, 150, 220).float().div(255)

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

class SignatureDataset(Dataset):
    def __init__(self, category, final_img_size, dim=(256, 256)):
        df = pd.read_csv(f'/opt/ml/input/data/meta/{category}_data.csv', header=0, names=['img_path_real', 'img_path_forged', 'label'])
        df['img_path_real'] = df['img_path_real'].apply(lambda x : f'/opt/ml/input/data/{category}/{x}')
        df['img_path_forged'] = df['img_path_forged'].apply(lambda x : f'/opt/ml/input/data/{category}/{x}')
        self.df  = df
        self.real_file_names = df["img_path_real"].values
        self.forged_file_names = df["img_path_forged"].values
        self.labels = df["label"].values
        self.dim = dim
        self.final_img_size=final_img_size

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self,index):
        real_file_path = self.real_file_names[index]
        forged_file_path = self.forged_file_names[index]
        
        img1 = load_image(real_file_path)
        img2 = load_image(forged_file_path)
        
        img1 = preprocess_image(img1, self.final_img_size, self.dim)
        img2 = preprocess_image(img2, self.final_img_size, self.dim)
        
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return torch.tensor(img1), torch.tensor(img2), label.float()
    
    
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)
    ))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--train-batch-size', 
                        type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', 
                        type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, 
                        default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, 
                        default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, 
                        default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = SignatureDataset('train', (1000, 1400))
    test_dataset = SignatureDataset('test', (1000, 1400))
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    criterion = nn.BCELoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    torch.save(model.state_dict(), "/opt/ml/model/siamese_network.pt")


if __name__ == '__main__':
    main()