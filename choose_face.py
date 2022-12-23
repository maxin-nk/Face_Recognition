# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
# !/usr/bin/env python
# -*-coding:utf8-*-
import os
import cv2
import time
import shutil


def getAllPath(dirpath, *suffix):
    """
    获取文件夹下各图片路径
    :param dirpath: 文件路径
    :param suffix:  图片格式
    :return:
    """
    PathArray = []

    # 遍历文件树
    for r, ds, fs in os.walk(dirpath):
        # 生成文件绝对路径
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray


def readPicSaveFace(sourcePath, targetPath, invalidPath, *suffix):
    """
    从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
    :param sourcePath: 源路径
    :param targetPath: 目标路径
    :param invalidPath: 未识别出脸的图片存放路径
    :param suffix: 图片格式
    :return:
    """
    try:
        # 获取图片绝对路径
        ImagePaths = getAllPath(sourcePath, *suffix)

        count = 1

        # 加载人脸识别分类器
        # haarcascade_frontalface_alt.xml为库训练好的分类器文件，下载opencv，安装目录中可找到
        face_cascade = cv2.CascadeClassifier(r'E:\a_test\haarcascade_frontalface_alt2.xml')

        for imagePath in ImagePaths:
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if type(img) != str:
                # 检测并返回图像中的人脸矩阵列表
                # 矩形区域左顶点对应x和y, 矩形区域的宽w, 矩形区域的高h
                faces = face_cascade.detectMultiScale(img, 1.01, 8)

                if len(faces):
                    for (x, y, w, h) in faces:
                        # 检查人脸像素，去除小于80像素的人脸
                        if w >= 80 and h >= 80:
                            # 生成文件名
                            listStr = [str(int(time.time())), str(count)]
                            fileName = '_'.join(listStr)  # s.join(iterator):以为分隔符链接迭代器中元素

                            # 扩大图片，可根据坐标调整
                            # img.shape[0]:height
                            # img.shape[1]:width
                            X = int(x * 0.5)
                            W = min(int((x + w) * 1.2), img.shape[1])
                            Y = int(y * 0.3)
                            H = min(int((y + h) * 1.4), img.shape[0])

                            # 重构矩形图片，并存储
                            f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                            cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                            count += 1
                            print(imagePath + "  have face")
                else:
                    # 将未检测到人脸的图片移出
                    shutil.move(imagePath, invalidPath)
    except IOError:
        print("Error")

    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)


if __name__ == '__main__':
    invalidPath = r'E:\a_test\haveNoPeople'
    sourcePath = r'E:\a_test\data'
    targetPath = r'E:\a_test\faceOfPeople'
    readPicSaveFace(sourcePath, targetPath, invalidPath, '.jpg', '.JPG', '.png', '.PNG')




