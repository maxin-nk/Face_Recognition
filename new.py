# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    获取数据集
    :param file_path: 数据集路径
    :return:
    """
    # 一个20 × 20的训练样本，共40个人，每人10张样本照片
    img = Image.open(file_path)
    img_ndarray = np.asarray(img, dtype='float64') / 256

    # 图片数据
    # 记录脸数据矩阵，57 * 47为每张脸的像素矩阵
    # 每张图片转换成一行数据
    faces = np.ones((400, 57 * 47))
    for i in range(20):
        for j in range(20):
            # 按照图片像素位置取对应图片数据
            faces[20 * i + j] = np.ndarray.flatten(img_ndarray[i * 57: (i + 1) * 57, j * 47: (j + 1) * 47])

    # 图片标签
    # one-hot编码
    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1

    # 数据集划分
    # 训练集，验证集，测试集
    train_data = np.zeros((320, 57 * 47))
    train_label = np.zeros((320, 40))
    vaild_data = np.zeros((40, 57 * 47))
    vaild_label = np.zeros((40, 40))
    test_data = np.zeros((40, 57 * 47))
    test_label = np.zeros((40, 40))

    for i in range(40):
        # 每个标签挑选前8作为训练集
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]
        # 每个标签第9张作为验证集
        vaild_data[i] = faces[i * 10 + 8]
        vaild_label[i] = label[i * 10 + 8]
        # 每个标签第10张作为测试集
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    vaild_data = vaild_data.astype('float32')
    test_data = test_data.astype('float32')

    return [(train_data, train_label), (vaild_data, vaild_label), (test_data, test_label)]


def convolutional_layer(data, kernel_size, bias_size, pooling_size):
    """
    卷积和池化
    :param data: 图像数组
    :param kernel_size: 卷积核大小
    :param bias_size: 偏差大小
    :param pooling_size: 池化大小
    :return:
    """
    # tf.random_normal_initializer: 生成正太分布的初始化器
    kernel = tf.get_variable("conv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias', bias_size, initializer=tf.random_normal_initializer())

    # 卷积
    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # relu激活函数转换
    linear_output = tf.nn.relu(tf.add(conv, bias))
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    return pooling


def linear_layer(data, weights_size, biases_size):
    """
    全连接层
    :param data:
    :param weights_size:
    :param biases_size:
    :return:
    """
    weights = tf.get_variable("weigths", weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_size, initializer=tf.random_normal_initializer())

    return tf.add(tf.matmul(data, weights), biases)


def convolutional_neural_network(data):
    """
    卷积神经网络
    :param data:
    :return:
    """
    # 输出层神经元个数
    n_output_layer = 40

    # 定义不同层的卷积核形状
    kernel_shape1 = [5, 5, 1, 32]
    kernel_shape2 = [5, 5, 32, 64]

    # 定义各层权重
    full_conn_w_shape = [15 * 12 * 64, 1024]
    out_w_shape = [1024, n_output_layer]

    # 定义各层误差
    bias_shape1 = [32]
    bias_shape2 = [64]
    full_conn_b_shape = [1024]
    out_b_shape = [n_output_layer]

    # -1：根据待转换数组自动计算
    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("conv_layer1") as layer1:
        layer1_output = convolutional_layer(data=data, kernel_size=kernel_shape1, bias_size=bias_shape1, pooling_size=[1, 2, 2, 1])

    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("conv_layer2") as layer2:
        layer2_output = convolutional_layer(data=layer1_output, kernel_size=kernel_shape2, bias_size=bias_shape2, pooling_size=[1, 2, 2, 1])

    with tf.variable_scope("full_connection") as full_layer3:
        # 将卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        layer3_output = tf.nn.relu(linear_layer(data=layer2_output_flatten, weights_size=full_conn_w_shape, biases_size=full_conn_b_shape))
        # layer3_output = tf.nn.dropout(layer3_output, 0.8)

    with tf.variable_scope("output") as output_layer4:
        output = linear_layer(data=layer3_output, weights_size=out_w_shape, biases_size=out_b_shape)

    return output


def train_facedata(dataset, model_dir, model_path):
    """
    模型训练
    :param dataset: 图像数组
    :param model_dir:
    :param model_path:
    :return:
    """
    # 一次投入模型的数据量
    batch_size = 40

    # 分别取得训练集、验证集和测试集数据
    train_set_x = dataset[0][0]
    train_set_y = dataset[0][1]
    valid_set_x = dataset[1][0]
    valid_set_y = dataset[1][1]
    test_set_x = dataset[2][0]
    test_set_y = dataset[2][1]

    X = tf.placeholder(tf.float32, [batch_size, 57 * 47])
    Y = tf.placeholder(tf.float32, [batch_size, 40])

    # 预测图片标签
    predict = convolutional_neural_network(X)

    # 梯度下降
    # 寻找最优模型，并保存当前节点状态
    lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(lost)
    saver = tf.train.Saver()

    # 定义会话
    with tf.Session() as session:
        # # 若不存在模型数据，需要训练模型参数
        # if not os.path.exists(model_path + ".index"):
        #     session.run(tf.global_variables_initializer())
        #     best_loss = float('Inf')
        #     for epoch in range(20):
        #         epoch_loss = 0
        #         for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
        #             x = train_set_x[i * batch_size: (i + 1) * batch_size]
        #             y = train_set_y[i * batch_size: (i + 1) * batch_size]
        #             _, cost = session.run([optimizer, lost], feed_dict={X: x, Y: y})
        #             epoch_loss += cost
        #
        #         print(epoch, ' : ', epoch_loss)
        #         if best_loss > epoch_loss:
        #             best_loss = epoch_loss
        #             if not os.path.exists(model_dir):
        #                 os.mkdir(model_dir)
        #                 print("create the directory: %s" % model_dir)
        #             save_path = saver.save(session, model_path)
        #             print("Model saved in file: %s" % save_path)
        #
        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        # 依次比较行最大值是否相同
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('模型精确度: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

        # 运用模型进行预测
        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})

        # 找出预测错误图像
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i' %(test_true[i], test_pred[i]))
        plot_errordata(incorrect_index, "olivettifaces.gif")


def plot_errordata(error_index, dataset_path):
    """
    画出在测试集中错误的数据
    :param error_index: 预测失误图像坐标
    :param dataset_path: 原始图像地址
    :return:
    """
    img = Image.open(dataset_path)
    plt.imshow(img)
    currentAxis = plt.gca()
    for index in error_index:
        row = index // 2
        column = index % 2
        currentAxis.add_patch(
            patches.Rectangle(
                xy=(
                     47 * 9 if column == 0 else 47 * 19,
                     row * 57
                    ),
                width=47,
                height=57,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            ))
    plt.savefig("result.png")
    plt.show()


def main():
    """
    主调用函数
    :return:
    """
    dataset_path = "olivettifaces.gif"
    data = load_data(dataset_path)
    model_dir = './model'
    model_path = model_dir + '/best.ckpt'
    train_facedata(data, model_dir, model_path)


if __name__ == "__main__" :
    main()

