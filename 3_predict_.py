import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image
from osgeo import gdal
gdal.UseExceptions()

import numpy as np
import torch
import time
import torch.nn.functional as F
from utils_new import load_data, preprocess_data

from conformer10 import Conformer
#from models import Conformer_tiny_patch16  # import model
import timm
from timm.models.registry import register_model
from timm.models.swin_transformer import _create_swin_transformer
from timm.models.swin_transformer import SwinTransformer


def predict(model, data_path, copy_data_path,save_path, device):
    """对完整数据集进行分类预测"""
    # 数据导入与预处理数据
    features, labels = load_data(data_path)
    features_tensor, labels_tensor = preprocess_data(features, labels)

    batch_size = 64
    full_dataset = TensorDataset(features_tensor, labels_tensor)
    full_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)

    # 预测
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in full_loader:
            inputs = F.pad(inputs, (0, 0, 7, 6, 7, 6), mode='constant', value=0)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs0,outputs1 = model(inputs)
            predicted = torch.max(outputs0+outputs1, dim=1)[1]
            predictions.extend(predicted.cpu().numpy())

    # 改变数值类型
    y_pred_img = np.array(predictions).astype(np.uint8)

    # 将预测结果转换为图像形状并且保存
    tag_g_ds = gdal.Open(copy_data_path)
    tag_g_data = tag_g_ds.ReadAsArray()
    print(f"Shape of tag_g_data: {tag_g_data.shape}")

    #tag_g_data_change = tag_g_data.transpose(1, 2, 0) 3维
    # No need to transpose since the array is already 2D
    tag_g_data_change = tag_g_data  # Directly use the 2D array

    height, width = tag_g_data_change.shape

    # Ensure y_pred_img matches the (height, width)
    y_pred_img = np.array(predictions).astype(np.uint8).reshape(height, width)

    # Convert to an image and save it
    image = Image.fromarray(y_pred_img)
    image.save(save_path)

   # height, width, channels = tag_g_data_change.shape
   # y_pred_img = y_pred_img.reshape(height, width)

    #image = Image.fromarray(y_pred_img)
    #image.save(save_path)

    return y_pred_img


if __name__ == '__main__':
    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    #model = Conformer_tiny_patch16(in_chans=3, num_classes=9).to(device)
    model = Conformer(in_chans=6, num_classes=9).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(r"F:\paper\CNN_frame\data\conformer\conformer10\pth\best_conformer_5_4.pth"))#权重路径

    # 进行预测0
    start_time = time.time()
    y_pred_img = predict(model,
                         r"F:\paper\CNN_frame\experiment\6band\h5\pyh_conformer_5_4.h5",#h5文件路径
                         r"F:\paper\CNN_frame\experiment\6band\disjoint\less\9_labels_sub_5_4.tif",#labels路径
                         r'F:\paper\CNN_frame\data\conformer\conformer10\output\pyh_conformer10_5_4.tif',#保存路径
                         device)
    end_time = time.time()
    print('Finished predict')
    spend_train = end_time - start_time # 计算时间差，得出的结果即为程序运行时间
    print("执行了{}秒".format(spend_train))