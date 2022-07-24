import os
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
from torch.utils import tensorboard

# 导入batch-size数据集
from flower_dataset import dataloaders

filename = 'checkpoint.pth'


"""""""""""""""冻结神经网络权重函数"""""""""""""""


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # 如果为True，就把迁移过来的网络模型各层权重梯度参数关闭
        # 但是在对网络的修改（下面函数中）
        # 在冻结所有层梯度之后又重写了最后层，所以最后输出层参数可以更新
        for param in model.parameters():
            param.requires_grad = False


"""""""""""""""修改内置模型全连接层"""""""""""""""


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    :param model_name: 模型名
    :param num_classes: 要进行分类的类别总数，也就是输出层节点个数
    :param feature_extract: 选择冻结那些层权重梯度参数，不进行更新
    :param use_pretrained: 是否使用ImageNet上的预训练模型
    :return:
    """
    # 选择合适的模型，不同模型初始化方法稍有区别
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        """ResNet152网络模型，选择使用ImageNet上的预训练模型"""
        model_ft = torchvision.models.resnet152(pretrained=use_pretrained)
        # 选择层权重梯度参数
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        """
        # fc, 是resnet类中定义的最后一层，即输出层(它的全连接层只有一层）
        # in_features 是linear(全连接层）的参数，表示输入数据的大小
        此处是，先获取到原本官方网络的输出层输入数据大小
        因为我们是在做迁移学习，前面部分连接地方不用动（包括参数）。
        最后面的输出层，要根据自己情况进行更改。
        但，输出层的输入数据是由不动的局部连接送过来的，所以in_features不变，改输出大小
        也就是改类别为自己的，102种花卉
        """
        model_ft.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1)
                                    )
        # 重新定义输出层。原本ResNet输出没有用激活函数，这里用了LogSoftmax(输出大小均为负数）
        input_size = 224  # 最初输入图像大小

    elif model_name == "alexnet":
        """ AlexNet模型"""
        model_ft = torchvision.models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        # 与之前类似。但是，AlexNet的全连接层有7层，此处要取第7层，获取他的输入数据大小参数
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # 重写第六层
        input_size = 224

    elif model_name == "vgg":
        """ VGG16"""
        model_ft = torchvision.models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        # VGG全连接部分同样是7层，取最后一层
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ SqueezeNet，具有AlexNet的精度，但是参数减少50倍。有两个版本，用第一个"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # 不同于AlexNet,SqueezeNet没有全连接层，最后输出部分也使用卷积层。
        # 所以重写它输出部分第2层（卷积层）
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ DenseNet"""
        model_ft = torchvision.models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        # 输出部分只有一个全连接层
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Inception V3模型要求输入数据为（N x 3 x 299 x 299）
        Be careful, expects (299,299) sized images and 
        has auxiliary output
        猜测，这里意思应该是说Inception模型中的分支，也就是辅助分类器
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 重写辅助分类器输出层
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Inception很特殊，有分支存在，分支辅助主干网络工作
        # 重写主干输出层
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299  # 特定的输入数据大小

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size  # 返回改写冻结后的网络模型和应该具有的输入数据尺寸


"""""""""""""""修改全连接层函数（官方）end"""""""""""""""


"""""""""""""""训练模型函数open"""""""""""""""
# 得到并保存神经网络模型checkpoint.pth
# 模型，数据，损失函数，优化器


def train_model(model, dataloaders, criterion,
                optimizer, num_epochs=25, is_inception=False,
                filename=filename):
    """
    :param model: 使用的模型
    :param dataloaders: 已处理好的batch数据
    :param criterion:
    :param optimizer: 优化器
    :param num_epochs: EPOCH次数
    :param is_inception: 是否使用Inception模型
    :param filename: 历史保存的文件名称
    :return:
    """
    since = time.time()
    best_acc = 0.0  # 存储模型对验证数据集的最高准确度
    """
    加载历史保存的模型参数
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    # para_groups是optimizer类中的一个参数，列表类型，包括两个字典元素
    # 其中第一个元素也就是param_groups[0]是优化器参数字典，包括各个参数对应的值
    # 其中'lr'就对应学习率

    # 最好的一次模型参数信息暂存
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 进入训练模式（dropout等层功能开启）
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0.0

            # 遍历数据集
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 通过上下文环境，设置一个局部开启梯度（更新参数）。启动词为"train"
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # Inception的辅助分类器，有两处分类输出。损失函数计算是按一定比例将辅助的加上
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, pred_s = torch.max(outputs, 1)

                    # 训练阶段更新权重，验证阶段无需
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算一个batch上的损失和验证正确的数量
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_s == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # 1个epoch的平均损失函数
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # 精确度（double()函数转换数据为float)

            time_elapsed = time.time() - since  # 一次epoch时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('现在为 {} 数据集， Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 每经过一个epoch，保存一次
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc  # 更新当前记录的验证数据集上最高精度值
                best_model_wts = copy.deepcopy(model.state_dict())
                # 更新当前学习到的最优模型参数
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()  # 将优化器的状态作为一个字典返回
                }
                torch.save(state, filename)  # 保存信息
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step()
                """
                optimizer.step()和scheduler.step()
                前者是优化器在每个batch学习中更新各个参数，
                而后者按官方定义是用来更新优化器学习率，一般是按照epoch为单位进行更新
                即多少个epoch后更换一次学习率，因而scheduler.step()一般放在epoch大循环下
                这里设置为7次epoch更新一下，也就是每7次epoch，scheduler.step()函数会被调用7次，再更新一次。
                但是，现在的lr_scheduler默认是，第0次（刚开始）就会更新一次
                所以，scheduler.step()一定是放在optimizer.step()后面
                """
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('当前optimizer lr: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])  # 记录学习率更新历史
        print()

    time_all_epoch = time.time() - since   # 总学习时间
    print('训练完成于 {:.0f}m {:.0f}s'.format(time_all_epoch // 60, time_all_epoch % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次作为最终结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


"""""""""""""""训练模型函数"""""""""""""""


"""""""""""""""加载并修改models中提供的ResNet模型"""""""""""""""
"""直接用训练的好权重当做初始化参数"""
# 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
model_name = 'resnet'

# 是否使用别人提取过的特征
feature_extract = True

# 判断设备是否可用
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 生成网络（改写输出层，冻结之前的层）
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

# 模型保存
filename = 'checkpoint.pth'

# 是否训练所有层（观察哪些层参数梯度属性是打开的即说明正在学习）
params_to_update = model_ft.parameters()
print("以下层参数正在学习：")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        # named_parameters()返回可迭代对象，包括参数名和值
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# 每7个epoch学习率衰减成原来的1/10
# 最后一层已经LogSoftmax()了，所以不用nn.CrossEntropyLoss()
# 因为nn.CrossEntropyLoss()相当于LogSoftmax()和nn.NLLLoss()结合（见官方文档）
# 定义损失函数
criterion = nn.NLLLoss()


"""""""""""""""在冻结前面层参数情况下，先训练自己的全连接层"""""""""""""""
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20, is_inception=(model_name == "inception"))
"""tensorboard可视化"""
writer = tensorboard.SummaryWriter("./train_model__only_linear")
model_list = [val_acc_history, train_acc_history, valid_losses, train_losses, LRs]
for idx_1, val_1 in enumerate(model_list):
    if idx_1 == 0:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("val_acc_history", val_2, global_step=idx_2)
    if idx_1 == 1:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("train_acc_history", val_2, global_step=idx_2)
    if idx_1 == 2:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("val_losses", val_2, global_step=idx_2)
    if idx_1 == 3:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("train_losses", val_2, global_step=idx_2)
    if idx_1 == 4:
        for idx_2, val_2 in enumerate(val_1):
            writer.add_scalar("LRs", val_2, global_step=idx_2)

writer.close()

"""""""""""""""自己的输出层训练完成后，再着手继续训练之前被冻结的层"""""""""""""""
for param in model_ft.parameters():
    param.requires_grad = True  # 将之前被冻结的参数“解冻”

optimizer = optim.Adam(params_to_update, lr=1e-4)   # lr学习率变小些
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.NLLLoss()

# 在之前训练好的输出层上再做训练
checkpoint = torch.load(filename)   # 加载之前保存的训练最好参数
best_acc = checkpoint['best_acc']   # 当前最好的一次准确率
model_ft.load_state_dict(checkpoint['state_dict'])   # 模型读取之前的参数
optimizer.load_state_dict(checkpoint['optimizer'])  # 优化器加载参数
# model_ft.class_to_idx = checkpoint['mapping']

# 调用函数，再训练一遍
model_ft, all_val_acc_history, all_train_acc_history, all_valid_losses, all_train_losses, all_LRs = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name =="inception"))
"""""""""""""""再训练所有层"""""""""""""""
writer2 = tensorboard.SummaryWriter("./train_model_all_layer")
model_list = [all_val_acc_history, all_train_acc_history, all_valid_losses, all_train_losses, all_LRs]
for idx_1, val_1 in enumerate(model_list):
    if idx_1 == 0:
        for idx_2, val_2 in enumerate(val_1):
            writer2.add_scalar("all_val_acc_history", val_2, global_step=idx_2)
    if idx_1 == 1:
        for idx_2, val_2 in enumerate(val_1):
            writer2.add_scalar("all_train_acc_history", val_2, global_step=idx_2)
    if idx_1 == 2:
        for idx_2, val_2 in enumerate(val_1):
            writer2.add_scalar("all_val_losses", val_2, global_step=idx_2)
    if idx_1 == 3:
        for idx_2, val_2 in enumerate(val_1):
            writer2.add_scalar("all_train_losses", val_2, global_step=idx_2)
    if idx_1 == 4:
        for idx_2, val_2 in enumerate(val_1):
            writer2.add_scalar("all_LRs", val_2, global_step=idx_2)
writer2.close()


"""""""""""""""测试网络效果open"""""""""""""""
"""probs, classes = predict ('flower_test.jpg', model_ft)  """
"""print(probs)                                            """
"""print(classes)                                          """
"""""""""""""""测试网络效果end"""""""""""""""
