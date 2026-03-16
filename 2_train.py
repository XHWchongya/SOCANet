import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import torch
# print(torch.version.cuda)
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchsummary import summary
import torch.optim as optim
import torch.nn.functional as F
from timm.utils import accuracy
import numpy as np
import torch
import h5py
from collections import Counter

import matplotlib.pyplot as plt
from utils_new import calculate_oa, calculate_kappa, calculate_confusion_matrix, load_data, preprocess_data, split_dataset, create_dataloaders

# 调用模型
from RMT import RMT_T3 # import model
#from vit_model import vit_base_patch16_224_in21k as vit
#from swintransformer_model import swin_tiny_patch4_window7_224 as swintransformer
#from MedMamba import VSSM as medmamba # import model
#from shufflenet_model import shufflenet_v2_x1_0
#from densenet_model import densenet121, load_state_dict
#from alexnet_model import AlexNet
#from vgg_model import vgg
#from resnet_model import resnet18
#from googlenet_model import GoogLeNet
#from efficientnet_model import efficientnet_b0 as efficientnetv1
#from efficientnetV2_model import  efficientnetv2_s as efficientnetv2
#from regnet_model import create_regnet as regnet
#from convnext_model import convnext_tiny as convnext


def train(model, train_loader, test_loader, device, num_epochs, criterion, optimizer,
          log_file=r'F:\paper\CNN_frame\data\RMT\txt\train_log_6_5.txt'):
    """
    训练模型，并将每个 epoch 的训练损失和准确率实时输出到控制台，并立即保存到文件中
    """
    train_losses = []
    best_accuracy = 0.0

    # 打开日志文件，准备实时记录
    with open(log_file, 'a') as f:  # 使用 'a' 模式确保不会覆盖文件内容
        # 如果文件为空，可以先写入标题
        f.write("Epoch, Train Loss, Test Accuracy\n")

        for epoch in range(num_epochs):
            # 训练过程
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 3, 1, 2)
                optimizer.zero_grad()
                output = model(inputs)
                # output0, output1 = model(inputs)

                # loss0 = criterion(output0, labels.long()) / 2
                # loss1 = criterion(output1, labels.long()) / 2
                # loss = loss0 + loss1
                # loss.backward()
                loss = criterion(output, labels.long())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 测试过程
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.permute(0, 3, 1, 2)
                    outputs = model(inputs)
                    # outputs0, outputs1 = model(inputs)

                    # predicted = torch.max(outputs0 + outputs1, dim=1)[1]
                    predicted = torch.max(outputs, dim=1)[1]
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

            # 输出当前 epoch 的损失和准确率
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Test Accuracy: {accuracy * 100:.2f}%')

            # 将当前 epoch 的结果写入日志文件
            f.write(f"{epoch + 1}, {avg_train_loss:.4f}, {accuracy * 100:.2f}%\n")

            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), r'F:\paper\CNN_frame\data\RMT\pth\best_RMT_6_5.pth')






if __name__ == '__main__':


    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_path =  r"F:\paper\CNN_frame\experiment\6band\h5\pyh_rmt_6_5.h5"


    # 数据导入与预处理数据
    features, labels = load_data(data_path)
    features_tensor, labels_tensor = preprocess_data(features, labels)


    # 划分训练集和测试集  不动
    X_train1, X_test1, y_train1, y_test1 = split_dataset(features_tensor, labels_tensor, train_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = split_dataset(X_train1,y_train1,  train_size=0.2, random_state=42)
    X_train_padded = F.pad(X_train, (0,0,7,6,7,6), mode='constant', value=0)
    # 对测试集图像进行 padding
    X_test_padded = F.pad(X_test, (0,0,7,6,7,6), mode='constant', value=0)
    print(X_train_padded.shape)
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(X_train_padded, y_train, X_test_padded, y_test, batch_size=64)  # 不优先调

    # 创建模型
    model = RMT_T3(in_chans=6, num_classes=9).to(device)
    # model = vit(img_size=19, in_chans=4, num_classes=8).to(device)  # 可以跑但是不稳定，需要参数调优
    # model = swintransformer(in_chans=4, num_classes=8).to(device)  # 可以跑但是不稳定，需要参数调优
    # model = medmamba(in_chans=4, num_classes=8).to(device) # false，需要推公式  # 一系列
    # model = shufflenet_v2_x1_0(in_chans=4, num_classes=8).to(device) # 一系列
    # model = densenet121(in_chans=4, num_classes=8).to(device) # 一系列
    # model = AlexNet(in_chans=4, num_classes=8).to(device) # padding or 网络层数
    # model = vgg(model_name="vgg13",in_chans=4, num_classes=8).to(device) # 一系列
    # model = resnet18(in_chans=4, num_classes=8).to(device) # 一系列
    # model = GoogLeNet(in_chans=4, num_classes=8).to(device)  # 一系列
    # model = efficientnetv1(in_chans=4, num_classes=8).to(device)  # false，需要推公式， # 一系列 懒得改了直接不用算了
    # model = efficientnetv2(in_chans=4, num_classes=8).to(device)
    # model = regnet(model_name="regnetx_200mf",in_chans=4, num_classes=8).to(device)  # 一系列
    # model = convnext(in_chans=4, num_classes=8).to(device)  # padding or 网络层数 # 一系列

    # 打印
    #summary(model, input_size=(4, 19, 19))


    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss().to(device)  # 这个里面包括了softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 不是越低越好  # 不优先调

    # 定义参数
    num_epochs = 50
    # 训练
    train(model, train_loader, test_loader, device,
          num_epochs, criterion, optimizer)

