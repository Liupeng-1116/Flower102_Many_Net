"""
识别102种花卉种类，输入一张花的照片，输出显示可能性最大的八种花名和该种花的照片。
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# 导入已经打包为batch的数据集和花名标签
from flower_dataset import dataloaders, cat_to_name

# 处理照片数据函数，检测照片预处理函数，展示一张照片函数
from flower_function import im_convert, process_image, imshow

"""""""""""""""
flower_model中在本程序需要用到的参数和函数本程序中重新写一遍
这样无需再调用flower_model程序，就不用再次训练模型"""""""""""""""
"""相关参数"""
feature_extract = True
model_name = 'resnet'
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 102),
            nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


"""""""""""""""生成测试所用的模型"""""""""""""""
# 加载模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)
filename = 'checkpoint.pth'
# 加载历史保存信息
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']  # 训练时的最高准确度
model_ft.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
# 这里无需再设置优化器和损失函数


"""""""""""""""设置用于检测的图像数据"""""""""""""""
image_path = './flower_test.jpg'
img1 = process_image(image_path)  # 预处理
imshow(img1)  # 显示该图片

# 得到一个batch的测试数据，在这里用模型进行检测
dataiter = iter(dataloaders['valid'])
# 生成一个可迭代对象
images, labels = dataiter.next()  # 不加next()则一直停留在第一个
model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())  # 图像tensor在GPU处理
else:
    output = model_ft(images)

# 得到概率最大的那个
_, pred_s_tensor = torch.max(output, 1)
# 这里加_,并不是错误。max()函数实际上返回的是一个元组，(max, max_idx)。
# 所以pred_s_tensor接受的是最大值索引
pred_s = np.squeeze(pred_s_tensor.numpy()) if not train_on_gpu else np.squeeze(pred_s_tensor.cpu().numpy())
# numpy不能接收GPU上的tensor，所以如果是在GPU上训练就要先将值转移到CPU,再转换为numpy类型


"""""""""""""""设置展示界面open"""""""""""""""
# 展示预测结果，这张照片最可能的前八类
fig = plt.figure(figsize=(20, 20))
columns = 4  # 2行4列
rows = 2

# 2*4 展示进行预测的8种花以及预测结果
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(pred_s[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(pred_s[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()

#  如果预测正确，标题颜色将会是绿色，否则为红色
