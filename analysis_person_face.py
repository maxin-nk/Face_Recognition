# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


def readCsv(file_train, file_test):
    img_train_data = []
    img_train_label = []
    with open(file_train, "r") as f:
        for img in f.readlines():
            # 获取训练集图片名
            img_train_data.append(img.strip().split(",")[0])
            # 获取训练集图片标签
            img_train_label.append(img.strip().split(",")[1])

    img_test_data = []
    img_test_label = []
    with open(file_test, "r") as f:
        for img in f.readlines():
            # 获取测试集图片名
            img_test_data.append(img.strip().split(",")[0])
            # 获取测试集图片标签
            img_test_label.append(img.strip().split(",")[1])
    return img_train_data, img_train_label, img_test_data, img_test_label


def readImg(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape
    img = cv2.resize(img, dsize=(M*N, 1))
    return np.array(img)


def main(length_train, length_test, img_train_data, img_train_label, img_test_data, img_test_label):
    result = []
    model = MLPClassifier(hidden_layer_sizes=(150,), activation="relu", learning_rate_init=0.0001, max_iter=300)
    # 神经网络模型训练
    for a in range(length_train):
        path = img_train_data[a]
        label = np.zeros([1, 41])
        num = int(img_train_label[a])
        label[0][num] = 1.0

        data = readImg(path)
        model.fit(data, label)
        print("train_processing:", a)

    # 模型预测
    for b in range(length_test):
        path_pre = img_test_data[b]
        label_real = img_test_label[b]
        data_pre = readImg(path_pre)
        res = model.predict(X=data_pre)
        result.append(res)
        print("real:%s   pre:%s" % (label_real, res))


if "__main__" == __name__:
    train, train_label, test, test_label = readCsv("train_list.csv", "test_list.csv")

    length_train = len(train)
    length_test = len(test)

    main(length_train, length_test, train, train_label, test, test_label)





