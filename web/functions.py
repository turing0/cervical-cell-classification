import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decimal import Decimal
from skimage import metrics, measure
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte


def pixel_wanted(pix):
    # return pix >= [0, 170, 0] and pix <= [80, 255, 80]
    # if pix[0] == 0 or pix[2] == 0:
    #     return True
    if pix[1] > pix[0] and pix[1] > pix[2]:
        return 'G'
    elif pix[2] > pix[1] and pix[2] > pix[1]:
        return 'B'
    else:
        return 'R'


def compute_shape_features(file):
    im = Image.open(file).convert('RGBA').convert('RGB')
    imnp = np.array(im)
    h, w = imnp.shape[:2]

    res = {
        'R': 0,
        'G': 0,
        'B': 0,
    }

    for x in range(h):
        for y in range(w):
            current_pixel = im.getpixel((x, y))
            res[pixel_wanted(current_pixel)] += 1
    nuclear_area = res['G']
    cell_area = h * w - res['B']
    # print(nuclear_area)

    return nuclear_area, cell_area


def compute_color_features(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    h, w, c = img.shape
    N = h * w

    R_t = img[:, :, 0]
    R = np.sum(R_t)
    R_2 = np.sum(np.power(R_t, 2.0))

    G_t = img[:, :, 1]
    G = np.sum(G_t)
    G_2 = np.sum(np.power(G_t, 2.0))

    B_t = img[:, :, 2]
    B = np.sum(B_t)
    B_2 = np.sum(np.power(B_t, 2.0))

    R_mean = R / N
    G_mean = G / N
    B_mean = B / N

    R_variance = R_2 / N - R_mean * R_mean
    G_variance = G_2 / N - G_mean * G_mean
    B_variance = B_2 / N - B_mean * B_mean

    R_mean = Decimal(R_mean).quantize(Decimal("0.000"))
    G_mean = Decimal(G_mean).quantize(Decimal("0.000"))
    B_mean = Decimal(B_mean).quantize(Decimal("0.000"))

    R_variance = Decimal(R_variance).quantize(Decimal("0.000"))
    G_variance = Decimal(G_variance).quantize(Decimal("0.000"))
    B_variance = Decimal(B_variance).quantize(Decimal("0.000"))

    return R_mean, G_mean, B_mean, R_variance, G_variance, B_variance


def compute_texture_features(file):
    img = io.imread(file)

    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                      normed=False, symmetric=False)

    contrast = greycoprops(matrix_coocurrence, 'contrast')
    energy = greycoprops(matrix_coocurrence, 'energy')
    asm = greycoprops(matrix_coocurrence, 'ASM')
    correlation = greycoprops(matrix_coocurrence, 'correlation')

    contrast = np.mean(contrast)
    energy = np.mean(energy)
    asm = np.mean(asm)
    correlation = np.mean(correlation)
    contrast = Decimal(contrast).quantize(Decimal("0.000"))
    energy = Decimal(energy).quantize(Decimal("0.000"))
    asm = Decimal(asm).quantize(Decimal("0.000"))
    correlation = Decimal(correlation).quantize(Decimal("0.000"))

    return energy, contrast, asm, correlation


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


# # img = cv2.imread("D:/lion.jpg", 0)
# # img = cv2.GaussianBlur(pil_img, (3, 3), 0)
# # canny = cv2.Canny(img, 50, 150)
# # plt.subplot(1, 2, 2)
# # plt.imshow(canny)
#
#
# print(output_label)
# # cv2.imshow('Canny', canny)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# img = cv2.imread('c.jpg')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 85, 230, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#
# # edges = cv2.Canny(img, 30, 100)
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()


def predict(file):
    # 加载模型
    net = HerlevCNet(3, 5)
    net = net.cuda()
    # model = TheModelClass(*args, **kwargs)
    net.load_state_dict(torch.load('web/net.pth'))
    net.eval()  # 在运行推理之前，务必调用model.eval()去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致 模型推断结果不一致

    image_path_list = []
    # for parent, dirnames, filenames in os.walk(r"C:\Users\Turing\Desktop\data"):
    #     for filename in filenames:
    #         image_path_list.append(os.path.join(parent, filename))
    image_path_list.append(file)
    dataset = ImageDataset(image_path_list)

    imgs, label, gt_image = dataset[0]

    imgs = imgs.cuda()
    output_label, output_gt, output_img = net(imgs.unsqueeze(0))
    pil_img = to_pil_image(output_img[0])
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    pil_img.save('output_images/' + 'output_img.jpg')
    # print('gt:',output_gt )
    # gt = to_pil_image(output_gt)
    # gt.save('gt.jpg')

    _, predicted = torch.max(output_label, 1)
    # print('dd---------------')
    # print(_, predicted.data)
    # print('dd---------------')

    # print(output_label)
    return output_label, predicted


def tensor_to_image(tensor):
    tensor = tensor.cpu().detach().numpy()*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)