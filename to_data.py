# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import pandas as pd
import numpy as np
import sys
import os


# ==========生成训练集数据
path = "train"
filenames = os.listdir(path)
strText = ""

with open(file="train_list.csv", mode="w") as f:
    # 打开父文件夹
    for a in range(len(filenames)):
        # 进一步打开子文件夹
        sub_path = path + os.sep + filenames[a]
        sub_filenames = os.listdir(sub_path)
        for i in range(len(sub_filenames)):
            # 获取图片的地址和标签
            strText = sub_path + os.sep + sub_filenames[i] + "," + filenames[a].split(sep="_")[1]+"\n"
            f.write(strText)


# ==========生成测试集数据
path = "test"
filenames = os.listdir(path)
strText = ""

with open(file="test_list.csv", mode="w") as f:
    for b in range(len(filenames)):
        strText = path + os.sep + filenames[b] + "," + filenames[b].split(sep="_")[1].split(sep=".")[0] + "\n"
        f.write(strText)

