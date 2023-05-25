import torchvision
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from skimage import metrics, measure
import cv2


class ImageDataset(Dataset):
    def __init__(self, image_path: list):
        super(ImageDataset, self).__init__()
        self.images = [x for x in image_path if '-d' not in x]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        origin_image = to_tensor(Image.open(self.images[index]).convert('RGB'))
        # label = torch.tensor(self.images[index].split('/')[-1][0], dtype=torch.long)
        label = self.images[index].split('\\')[-1][0]
        gt_image = (Image.open(self.images[index].replace('.BMP', '-d.bmp')))
        return origin_image, label, gt_image


'''
        origin_image = (np.array(Image.open(self.images[index])))
        origin_image = np.transpose(origin_image,[2,0,1])
        origin_image = torch.tensor(origin_image)
'''


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)


class TripleConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            # nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)


class Upsamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsamples, self).__init__()
        self.unpool = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        a = self.unpool(x)
        return a


class HerlevCNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256]):
        super(HerlevCNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)  # stride=2 ?
        self.upsample = Upsamples(64, 64)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.conv11 = nn.Conv2d(128, 64, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 7)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        self.conv = nn.Conv2d(64, 5, 3, padding=1)
        self.conv_final = nn.Conv2d(5, 3, 3, padding=1)

        # Down part of HerlevCNet
        for index, feature in enumerate(features):
            if index == 2:
                self.downs.append(TripleConv(in_channels, feature))
            else:
                self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of HerlevCNet
        for index, feature in enumerate(reversed(features[1:])):
            # self.ups.append(nn.MaxUnpool2d(kernel_size=2))
            # print(feature)
            self.ups.append(Upsamples(feature, feature))
            if index == 0:
                self.ups.append(TripleConvUp(feature, feature // 2))
            else:
                self.ups.append(DoubleConvUp(feature, feature // 2))

    def forward(self, x):
        skip_connections = []
        pooling_indices = []

        for idx in range(0, len(self.downs)):

            x = self.downs[idx](x)
            # print(x.shape)
            if idx == 0:
                skip_connections.append(x)
            x, indice = self.pool(x)
            pooling_indices.append(indice)

        y = self.gap(x)
        y = self.flatten(y)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        # print('y:', y)
        label = y
        pooling_indices = pooling_indices[::-1]

        for idx in range(0, len(self.ups), 2):
            # print("ups:", x.shape)
            x = self.ups[idx](x)
            x = self.ups[idx + 1](x)

        # x = self.unpool(x, pooling_indices[-1])

        x = self.upsample(x)
        # print("afterunsample:",x.shape)
        cancat_skip = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv11(cancat_skip)
        # print("cancat_skip:", x.shape)
        x = self.conv(x)
        x = self.relu(x)
        gt_loss = self.softmax(x)
        x_for_img = self.conv_final(x)
        x_for_img = self.softmax(x_for_img)
        return label, gt_loss, x_for_img
        # return label


# 加载模型
net = HerlevCNet(3, 5)
net = net.cuda()
# model = TheModelClass(*args, **kwargs)
net.load_state_dict(torch.load('net.pth'))
net.eval()  # 在运行推理之前，务必调用model.eval()去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致 模型推断结果不一致

image_path_list = []
for parent, dirnames, filenames in os.walk('C:\Brec\data'):
    for filename in filenames:
        image_path_list.append(os.path.join(parent, filename))
dataset = ImageDataset(image_path_list)
imgs, label, gt_image = dataset[185]
uimg = to_pil_image(imgs)
imgs = imgs.cuda()
output_label, output_gt, output_img = net(imgs.unsqueeze(0))
pil_img = to_pil_image(output_img[0])
plt.subplot(1, 2, 1)
plt.imshow(pil_img, cmap='gray')
pil_img.save('c.jpg')
# img = cv2.imread("D:/lion.jpg", 0)
# img = cv2.GaussianBlur(pil_img, (3, 3), 0)
# canny = cv2.Canny(img, 50, 150)
# plt.subplot(1, 2, 2)
# plt.imshow(canny)


print(output_label)
# cv2.imshow('Canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = cv2.imread('c.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 85, 230, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# edges = cv2.Canny(img, 30, 100)
plt.subplot(1, 2, 2)
plt.imshow(uimg, cmap='gray')
plt.show()
