# 定义了一些处理图像的函数，包括数据集预处理，图片展示等
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


filename = 'checkpoint.pth'


"""--------------数据集预处理操作--------------
# 图像增强：将数据集中照片进行旋转、翻折、放大...得到更多的数据
"""

data_transforms = {
    'train':
        transforms.Compose([
            transforms.RandomRotation(45),   # 随机旋转，旋转角度在-45到45度之间随机
            transforms.CenterCrop(224),   # 中心裁剪，图片大小为224*224
            transforms.RandomHorizontalFlip(p=0.5),   # 50%概率随机水平翻转图像
            transforms.RandomVerticalFlip(p=0.5),    # 50%概率随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 调整图片亮度，对比度，饱和度，色相
            transforms.RandomGrayscale(p=0.025),
            # 0.025概率转换成灰度图像，输入是3通道，输出就是R=G=B（仍然3通道）
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # 按通道标准化
            ]),
    'valid':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
"""""""""""""""数据集预处理"""""""""""""""


"""---------------处理照片数据函数--------------"""
# 注意tensor的数据需要转换成numpy的格式，而且还需要还原回标准化的结果


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    # 首先，numpy不支持GPU上的tensor,所以首先将tensor转移到CPU
    # clone一份数据后，进行detach(),返回一个tensor(不带梯度信息）
    image = image.numpy().squeeze()
    # 对CPU上tensor进行numpy()格式转换，并且删除掉所有为1的维度。
    image = image.transpose(1, 2, 0)  # 换轴
    # （C, H, W)还原回(H，W, C)格式
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    # 按照标准化时计算方式的逆运算，还原非标准化数据
    image = image.clip(0, 1)
    # numpy.clip(self, min=, max=, out=)  控制self数组内的元素大小全部在min和max范围内
    # 本就在范围内的不变，小于的变为min，大于的变为max
    return image


"""""""""""""""处理照片数据函数"""""""""""""""


"""""""""""""""检测照片预处理函数"""""""""""""""


def process_image(image_path):
    # 读取测试图像
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
        # 制作图像缩略图，内部调用了resize函数
        # 可认为是对resize的封装，但是thumbnail不会改变图片原有比例
    else:
        img.thumbnail((256, 10000))
    # Crop操作，再裁剪
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    # 左右上下边距
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))  # 按各个边距裁剪
    # 相同的预处理方法
    img = np.array(img) / 255  # 归一化像素值，同时除以最大值255
    mean = np.array([0.485, 0.456, 0.406])  # 提供的均值
    std = np.array([0.229, 0.224, 0.225])  # 提供的标准差
    img = (img - mean) / std  # 标准化

    img = img.transpose((2, 0, 1))  # 换轴，np.array(img)转换默认HWC格式
    # (H, W, C)转换（C, H, W)
    return img


"""""""""""""""检测照片预处理函数end"""""""""""""""


"""""""""""""""展示一张照片函数"""""""""""""""


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))
    # 还原为HWC

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean  # 标准化逆运算
    image = np.clip(image, 0, 1)  # 限定元素值上下限

    ax.imshow(image)  # 在窗口展示一张经过上述还原操作的图像
    ax.set_title(title)  # 标题
    plt.show()
    return ax


"""""""""""""""展示一张照片函数end"""""""""""""""

