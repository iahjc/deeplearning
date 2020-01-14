import numpy as np

if __name__ == "__main__":
    print(np.__version__)
    lst = [1, 2, 3]
    print(lst)

    vec = np.array([1, 2, 3])
    # vec[0] = '2321aaa'
    print(vec)

    print(np.zeros(5))

    print(np.ones(5))

    print(np.full(5, 66.))

    # np.array的基本属性
    print("size = ", vec.size)
    print("size = ", len(vec))

    # 切片语法
    print(vec[0: 2])

    vec2 = np.array([4, 5, 6])
    print("{} + {} = {}".format(vec, vec2, vec + vec2))
    print("{} - {} = {}".format(vec, vec2, vec - vec2))

    print("{} * {} = {}".format(2, vec, 2 * vec))

    print("{}.dot({}) = {}".format(vec, vec2, vec.dot(vec2)))

    print(np.linalg.norm(vec))