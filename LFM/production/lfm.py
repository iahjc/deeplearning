#-*-coding:utf8-*-
"""
author: alines
date: 2020
lfm model train main function
"""
import numpy as np
import sys
sys.path.append("../util")
import util.read as read
def lfm_train(train_data, F, alpha, beta, step):
    """
    :param train_data:
            train_data for lfm
    :param F:
            user vector len, item vector len
    :param alpha:
            regularization factor
    :param beta:
            learning rate
    :param step:
            iteration num
    :return:
            dict: key itemid, value:list
            dict: key userid, value:list
    """

    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        for data_instance in train_data:
            userid, itemid, label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
        delta = label - model_predict(user_vec[userid], item_vec[itemid])
        for index in range(F):
            user_vec[userid][index] += beta *(delta*item_vec[itemid][index] - alpha*user_vec[userid][index])
            item_vec[itemid][index] += beta*(delta*user_vec[itemid][index] - alpha*item_vec[itemid][index])
        beta = beta * 0.9
    return user_vec, item_vec

def init_model(vector_len):
    """
    :param vector_len: the len of vector
    :return: a ndarray
    """
    return np.random.randn(vector_len)

def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance
    :param user_vector: model produce user vector
    :param item_vector:  model produce item vector
    :return: a num
    """
    res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res


def model_train_process():
    """
    test lfm model train
    :return:
    """
    train_data = read.get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)


if __name__ == "__main__":
