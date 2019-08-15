import numpy as np
# 科学计算的基本包
import matplotlib.pyplot as plt
# python 图形库
import h5py
import os
# 是与存储在h5文件中的数据集交互的通用包
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# dataset
# 1. 标记为猫是1 不是猫为0 的训练图像集
# 2. 标记为猫或非猫的测试训练图像集
# 3. 每个图形的形状(num_px, num_px, 3),其中3表示3个通道(RGB). 因此每个图像都是正方形(高度=num_px) 和 (宽度=num_px)


# 构建一个简单的图像识别算法，可以正确地将图片分类为猫或非猫。
# https://blog.csdn.net/u013733326/article/details/79639509 课后练习题

# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# dir_path = os.path.dirname(os.path.abspath(__file__))
# print('当前目录绝对路径:', dir_path)

index = 25
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + " picture")
print("==========================================================")
m_train = train_set_y.shape[1] # 训练集里图片的数量
m_test = test_set_y.shape[1] # 测试集里图片的数量
num_px = train_set_x_orig.shape[1] # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看看我们加载的东西的具体情况
print ("训练集的数量：m_train=" + str(m_train))
print ("测试集的数量：m_test=" + str(m_test))
print ("每张图片的宽/高：num_px=" + str(num_px))
print ("每张图片的大小：(" + str(num_px) + "," + str(num_px) + ", 3)")
print("训练集_图片的维数："+str(train_set_x_orig.shape))
print("训练集_标签的维数：" + str(train_set_y.shape))
print("测试集_图片的维数：" + str(test_set_x_orig.shape))
print("测试集_标签的维数" + str(test_set_y.shape))

print("==============================================================")

# 将训练集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("训练集降维最后的维度：" + str(train_set_x_flatten.shape))
print("训练集_标签的维数：" + str(train_set_y.shape))
print("测试集降维之后的维度：" + str(test_set_x_flatten.shape))
print("测试集_标签的维数：" + str(test_set_y.shape))

print("================================================================")

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# 建立神经网络的主要步骤是：
# 1. 定义模型结构（例如输入特征的数量）
# 2. 初始化模型的参数
# 3. 更新参数（梯度下降）

def sigmoid(z):
    """
    :param z: 任意大小的标量或numpy数组
    :return: s - sigmoid (z)
    """
    s = 1 / (1 + np.exp(-z))
    return s

# 测试sigmoid
print("==========================================")
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(9.2) = " + str(sigmoid(9.2)))


# 既然sigmoid测试好了，我们现在就可以初始化我们需要的参数w和b了。
def initialize_with_zeros(dim):
    """
    此函数为w创建一个维度为(dim， 1)的0向量，并将b初始化为0
    :param dim: 我们想要的w矢量的大小（或者这种情况下参数的数量）
    :return:
        w - 维度为(dim，1)的初始化向量
        b - 初始化标量（对应于偏差）
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    # 使用断言来确保我们的数据是正确的
    assert (w.shape == (dim, 1)) # w的维度是(dim, 1)
    assert (isinstance(b, float) or isinstance(b, int)) # b的类型是float 或者int
    return (w, b)


# 我们现在要实现一个计算成本函数及其渐变的函数propagate（）
def propagate(w, b, X, Y):
    """
    实现向前和向后传播的成本函数以及梯度
    :param w: 权重，大小不等的数组(num_px * num_px * 3, 1)
    :param b: 偏差，一个标量
    :param X: 矩阵类型为(num_px*num_px*3, 训练数量)
    :param Y: 真正的标签矢量（如果非猫则为0，如果是猫则为1）， 矩阵维度为（1， 训练数据数量）
    :return:
        cost - 逻辑回归的负对数似然成本
        dw - 相对于w的损失梯度，因此与w相同的形状
        db - 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b) # 计算激活值
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) # 计算成本

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，把dw和db保存起来。
    grads = {
        "dw": dw,
        "db": db
    }
    return (grads, cost)

#测试一下propagate
print("====================测试propagate====================")
#初始化一些参数

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):
    """
    此函数通过运行梯度下降算法来优化w和b

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值

    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))

    params  = {
                "w" : w,
                "b" : b }
    grads = {
            "dw": dw,
            "db": db }
    return (params , grads , costs)

#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w , b , X ):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

    """

    m  = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T , X) + b)
    for i in range(A.shape[1]):
        #将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))

    return Y_prediction

#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))

def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型

    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本

    返回：
        d  - 包含有关模型信息的字典。
    """
    w , b = initialize_with_zeros(X_train.shape[0])

    parameters , grads , costs = optimize(w , b , X_train , Y_train,num_iterations , learning_rate , print_cost)

    #从字典“参数”中检索参数w和b
    w , b = parameters["w"] , parameters["b"]

    #预测测试/训练集的例子
    Y_prediction_test = predict(w , b, X_test)
    Y_prediction_train = predict(w , b, X_train)

    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")

    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d

print("====================测试model====================")
#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

