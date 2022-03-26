import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from .functions import tensor_to_image
import cv2
import matplotlib.pyplot as plt
from skimage import metrics, measure
import cv2
from skimage.feature import greycomatrix, greycoprops
from .models import User


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
        # gt_image = (Image.open(self.images[index].replace('.BMP', '-d.bmp')))
        # return origin_image, label, gt_image
        return origin_image, label, 0


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


class ChurnModel:
    """ Wrapper for loading and serving pre-trained model"""

    def __init__(self):
        # self.model = self._load_model_from_path(MODEL_PATH)

        self.model = HerlevCNet(3, 5)
        self.model = self.model.cuda()
        # model = TheModelClass(*args, **kwargs)
        self.model.load_state_dict(torch.load('web/net.pth'))
        self.model.eval()

    def predict(self, file, user_headimg):
        image_path_list = []
        image_path_list.append(file)
        dataset = ImageDataset(image_path_list)

        imgs, label, gt_image = dataset[0]

        imgs = imgs.cuda()
        output_label, output_gt, output_img = self.model(imgs.unsqueeze(0))
        pil_img = to_pil_image(output_img[0])
        # plt.subplot(1, 2, 1)
        # plt.imshow(pil_img)
        pil_img.save('output_images/' + 'output_img.jpg')

        # img = Image.open('output_images/' + 'output_img.jpg').convert("L")  # 读图片并转化为灰度图
        img = pil_img.convert("L")  # 读图片并转化为灰度图
        img_array = np.array(img)  # 转化为数组
        w, h = img_array.shape
        img_border = np.zeros((w - 1, h - 1))
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                Sx = img_array[x + 1][y - 1] + 2 * img_array[x + 1][y] + img_array[x + 1][y + 1] - \
                     img_array[x - 1][y - 1] - 2 * \
                     img_array[x - 1][y] - img_array[x - 1][y + 1]
                Sy = img_array[x - 1][y + 1] + 2 * img_array[x][y + 1] + img_array[x + 1][y + 1] - \
                     img_array[x - 1][y - 1] - 2 * \
                     img_array[x][y - 1] - img_array[x + 1][y - 1]
                img_border[x][y] = (Sx * Sx + Sy * Sy) ** 0.5

        img2 = Image.fromarray(img_border)
        # img2.show()
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
            # img2.save('output_images/' + 'segmentation_result.jpg')
            segmentation_img_name = str(user_headimg).replace('.BMP', '_segmentation_img.jpg')   #   img/xxxxxxxx.BMP
            img2.save('media/' + segmentation_img_name)

            # segmentation_img = User.objects.create(headimg='output_images/' + 'segmentation_result.jpg')
            # user = User(headimg='output_images/' + 'segmentation_result.jpg')
            # segmentation_img.save()

            # print('------segmentation_result---------')
            # print('segmentation_result: ', segmentation_img.headimg)


        # print('shape:', output_gt.shape, output_img.shape)
        # gt_img = to_pil_image(output_gt)
        # gt_image = tensor_to_image(output_gt)

        # gt_image.save('output_images/' + 'gt_image.jpg')

        _, predicted = torch.max(output_label, 1)
        return output_label, predicted, segmentation_img_name
