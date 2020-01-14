import numpy as np

my_array = np.array([1, 2, 3, 4, 5])

print (my_array)

print(my_array.shape)

print(my_array[0])

print(my_array[1])

my_array[0] = -1

print(my_array)

# 创建一个长度为5的的Numpy数组，但所有的元素为0
my_new_array = np.zeros((5))
print(my_new_array)

# 创建一个随机数组
my_random_array = np.random.random((5))
print(my_random_array)

# 创建二维数组
my_2d_array = np.zeros((2, 3))
print(my_2d_array)

# 创建都为1 的 两行四列的二维数组
my_2d_array_new = np.ones((2, 4))
print(my_2d_array_new)

my_array = np.array([[4, 5], [6, 1]])
print(my_array[1][0])

print(my_array.shape)

my_array_column_2 = my_array[:, 1]
print(my_array_column_2)


a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
sum = a + b
difference = a - b
product = a * b
quotient = a / b
print("Sum = \n" , sum)
print("defference = \n", difference)
print("product = \n", product)
print("quotient = \n", quotient)


# 矩阵乘法
matrix_product = a.dot(b)
print("Matrix Product = \n", matrix_product)



# 10 Array
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.linspace(0, 2 * np.pi, 5)

print(a)
print(b)
print(c)
print(d)