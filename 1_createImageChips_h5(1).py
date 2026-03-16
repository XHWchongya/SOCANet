import os, math, random, glob, time
import numpy as np
import h5py
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile
from sklearn.model_selection import train_test_split

# define the file names 打开影像和真值
feature_file = r"F:\paper\CNN_frame\experiment\6band\disjoint\less\9_s12_sub_6_7.tif"
label_file = r"F:\paper\CNN_frame\experiment\6band\disjoint\less\9_labels_sub_6_7.tif"
# create feature chips using pyrsgis 切片
features = imageChipsFromFile(feature_file, x_size=19, y_size=19)#读取输入图片 并切片为3x3
features=features/features.max()

# read the label file and reshape it
ds, labels = raster.read(label_file)#读取特征文件并展开
labels = labels.flatten()
#size,x,y,bands=features.shape
#X_index = np.arange(0, size)
# Passing index array instead of the big feature matrix
#X_train, X_test, train_labels, test_labels = train_test_split(X_index, labels, train_size=0.01, random_state=150)
#train_data = features[X_train,:,:,:]
#print("train_data.shape:",train_data.shape)
#test_data = features[X_test,:,:,:]
#print("test_data.shape:",test_data.shape)
#np.save(r"D:\NPY\S1\S1train_x.npy",train_data)
#np.save(r"D:\NPY\S1\S1test_x.npy",test_data)
#np.save(r"D:\NPY\S1\S1train_y.npy",train_labels)
#np.save(r"D:\NPY\S1\S1test_y.npy",test_labels)

# 读取
# h5f = h5py.File(r'E:\NPY\S12\S12data23.h5','r')
# features= np.asarray(h5f['features'])
# labels= np.asarray(h5f['labels'])
# #train_x= np.asarray(h5f['train_x'])
# #test_x= np.asarray(h5f['test_x'])
# #train_y= np.asarray(h5f['train_y'])
# #test_y= np.asarray(h5f['test_y'])
# h5f.close()

# print basic details 输出大小、最大值最小值
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
# print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))
# Save the arrays as .npy files 保存为npy文件
#np.save(r'D:\NPY\S1\CNN_S1_features.npy', features)#保存为npz shape数组文件
#np.save(r'D:\NPY\S1\CNN_S1_labels.npy', labels)
print('Arrays saved at location %s' % (os.getcwd()))

h5f = h5py.File(r'F:\paper\CNN_frame\experiment\6band\h5\pyh_conformer_6_7.h5', 'w')
#h5f.create_dataset('train_x', data=train_data)
#h5f.create_dataset('train_y', data=train_labels)
#h5f.create_dataset('test_x', data=test_data)
#h5f.create_dataset('test_y', data=test_labels)
h5f.create_dataset('features', data=features)
h5f.create_dataset('labels', data=labels)
h5f.close()
