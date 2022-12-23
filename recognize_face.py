# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def load_data(file_path):
    """
    加载训练集数据
    :param file_path: 文件路径
    :return: 数据和标签
    """
    img_address = []
    img_label = []
    # 获取图片的数据与标签
    with open(file_path, "r") as f:
        for a in f.readlines():
            img_address.append(a.strip().split(",")[0])
            img_label.append(a.strip().split(",")[1])

    # 记录脸矩阵
    faces = np.zeros((400, 92*112))
    labels = np.zeros((400, 40))
    # labels = np.zeros((400,))
    for b in range(len(img_address)):
        img = Image.open(img_address[b])
        img = np.asarray(img, dtype='float64') / 255
        # 转换成一维数组
        faces[b][:] = img.flatten()
        faces[b][:] /= 255
        num = int(img_label[b])
        labels[b][num] = 1.0
        # labels[b] = img_label[b]

    return faces, labels


def load_test(file_path):
    """
    加载测试集数据
    :param file_path:
    :return:
    """
    img_address = []
    img_label = []

    with open(file_path, "r") as f:
        for a in f.readlines():
            img_address.append(a.strip().split(",")[0])
            img_label.append(a.strip().split(",")[1])

    # 记录脸矩阵
    faces = np.zeros((40, 92*112))
    labels = np.zeros((40, 40))
    # labels = np.zeros((40, ))
    for b in range(len(img_address)):
        img = Image.open(img_address[b])
        img = np.asarray(img, dtype=np.float64) / 255
        # 转换成一维数组
        faces[b][:] = img.flatten()
        faces[b][:] /= 255
        num = int(img_label[b])
        labels[b][num] = 1.0
        # labels[b] = img_label[b]

    return faces, labels



def model_train(file_path, model):
    """
    模型训练
    :param file_path: 训练集路径
    :param model: 模型
    :return: 训练后模型
    """
    train, train_label = load_data(file_path)
    M, N = train.shape
    # 神经网络模型训练
    for a in range(M):
        # model.fit(np.array(train[a][:]), train_label[a][:])
        model.fit(np.array(train[a][:]).reshape(1, -1), np.array(train_label[a]))
        print("train_processing:", a)

    return model


def model_predict(file_path, model):
    """
    模型预测
    :param file_path: 测试集路径
    :param model: 训练后模型
    :return: 真实值\预测值
    """
    result = np.zeros((400, 1))
    test, test_label = load_test(file_path)
    # 模型预测
    M, N = test.shape
    for b in range(M):
        # res = model.predict([test[b][:]])
        res = model.predict(np.array(test[b][:]))
        # result[b][0] = res
        print("real:%s   pre:%s" % (test_label[b][:], res))
    return test_label, result


if "__main__" == __name__:
    # model = MLPClassifier(hidden_layer_sizes=(50,), activation="relu", learning_rate_init=0.01, max_iter=25)
    model = SVC()
    # model.fit()
    model = model_train("train_list.csv", model)

    real_label, pre_label = model_predict("test_list.csv", model)

























