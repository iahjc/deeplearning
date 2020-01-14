import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # 设置一个固定的随机种子，以保证接下来步骤中我们的结果是一致的

X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c = Y, s=40, cmap=plt.cm.Spectral) # 绘制散点图
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图

# 我们来仔细观察这些数据
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1] # 训练集里面的数量

print("X的维度为" + str(shape_X))
print("Y的维度为" + str(shape_Y))
print("数据集里面的数据有：" + str(m) + "个")


# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)

# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
#         np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")


"""
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）

    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
"""
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


# 测试layer_sizes
#测试layer_sizes
print("=========================测试layer_sizes=========================")
X_asses , Y_asses = layer_sizes_test_case()
(n_x,n_h,n_y) =  layer_sizes(X_asses,Y_asses)
print("输入层的节点数量为: n_x = " + str(n_x))
print("隐藏层的节点数量为: n_h = " + str(n_h))
print("输出层的节点数量为: n_y = " + str(n_y))

# 初始化模型参数
def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x: 输入层节点的数量
    :param n_h: 隐藏层节点的数量
    :param n_y: 输出层节点的数量
    :return:
        W1 权重矩阵，维度为(n_h, n_x)
        b1 偏向量，维度为(n_h, 1)
        W2 权重矩阵，维度为(n_y, n_h)
        b2 偏向量，维度为(n_y, 1)
    """
    np.random.seed(2) # 指定一个随机种子，以便您的输出和我们的一样
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言确保我的数据格式是正确的
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }

    return parameters

#测试initialize_parameters
print("=========================测试initialize_parameters=========================")
n_x , n_h , n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x , n_h , n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))