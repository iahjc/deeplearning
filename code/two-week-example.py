import numpy as np
# 科学计算的基本包
import matplotlib.pyplot as plt
# python 图形库
import h5py
# 是与存储在h5文件中的数据集交互的通用包
import scipy
from PIL import Image
from scipy import ndimage
# from lr_utils import load_dataset

# dataset
# 1. 标记为猫是1 不是猫为0 的训练图像集
# 2. 标记为猫或非猫的测试训练图像集
# 3. 每个图形的形状(num_px, num_px, 3),其中3表示3个通道(RGB). 因此每个图像都是正方形(高度=num_px) 和 (宽度=num_px)