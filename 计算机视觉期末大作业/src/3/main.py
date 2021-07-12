import cv2
import numpy as np
import math
from segmentation import *
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class position:
    def __init__(self, height, width):
        self.t = np.zeros(shape=(height*width+width), dtype=tuple)
        self.height = height
        self.width = width
        for y in range(height):
            for x in range(width):
                self.t[y*width+x] = (y, x)


# 提取归一化RGB颜色直方图
def cal_hist(img, bin, mask=None):
    # 若没有遮罩，则将图像变为(height*width,3)的一列像素
    if mask is None:
        height, width, _ = img.shape
        img = np.reshape(img, newshape=(height*width, 3))
    else:
        # 拉伸
        assert img.shape[:2] == mask.shape
        front_size = len(mask[mask == 255])
        ret = np.zeros(shape=(front_size, 3), dtype=np.uint8)
        height, width, _ = img.shape
        i = 0
        for r in range(height):
            for c in range(width):
                if mask[r, c] == 255:
                    ret[i] = img[r, c]
                    i += 1
        img = ret
    length, channel = img.shape
    assert channel == 3
    interval = 256 / bin
    colorspace = np.zeros(shape=(bin, bin, bin), dtype=float)
    for p in range(length):
        v = img[p, :]
        i, j, k = math.floor(v[0]/interval), math.floor(v[1]/interval), math.floor(v[2]/interval)
        colorspace[i, j, k] += 1
    res = np.reshape(colorspace, newshape=int(math.pow(bin, 3)))
    res = res / length
    return res


# 计算特征矩阵
def cal_fmat(img, bin, res, t):
    # 提取归一化RGB颜色直方图
    img_fvec = cal_hist(img, bin)
    m = []
    # 图像分割后的每一个区域
    for comp in res.all_area():
        mask = np.zeros(shape=(t.height, t.width), dtype=np.uint8)
        v = res.one_area(comp)
        for i in v:
            pix = t.t[i]
            mask[pix[0], pix[1]] = 255
        comp_fvec = cal_hist(img, bin, mask)
        # 拼接颜色直方图和全图颜色直方图
        fvec = np.concatenate((comp_fvec, img_fvec))
        m.append(fvec)
    m = np.array(m)
    return m


def cal_ytrain(t, comps, img_mark):
    y_train = []
    for comp in comps:
        (y, x) = t.t[comp]
        if img_mark[y, x] == 255:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train


def data_generate():
    x_train, y_train = [], []
    x_test, y_test = [], []
    # 提取特征
    for i in range(1, 200):
        img_number = str(i*4)
        # 排除末尾是学号的
        if((i*4) % 100 != 52):
            # 打印进度
            print(str(i)+" of 199")
            # 读取相应编号训练原图和前景图
            img = cv2.imread("train/img/"+img_number+"_origin.png")
            img_mark = cv2.imread("train/mark/"+img_number+"_mark.png")
            img_mark = cv2.cvtColor(img_mark, cv2.COLOR_BGR2GRAY)
            # 调用之前写的图像分割
            res = segment(img)
            t = position(img.shape[0], img.shape[1])
            # 计算特征矩阵
            fmat = cal_fmat(img, 8, res, t)
            for fvec in fmat:
                x_train.append(fvec)
            y_train = y_train + cal_ytrain(t, res.all_area(), img_mark)
        # 末尾是学号
        else:
            # # 读取相应编号测试原图和前景图进行相同操作
            img = cv2.imread("test/img/"+img_number+"_origin.png")
            img_mark = cv2.imread("test/mark/"+img_number+"_mark.png")
            img_mark = cv2.cvtColor(img_mark, cv2.COLOR_BGR2GRAY)
            res = segment(img)
            t = position(img.shape[0], img.shape[1])
            fmat = cal_fmat(img, 8, res, t)
            for fvec in fmat:
                x_test.append(fvec)
            y_test = y_test + cal_ytrain(t, res.all_area(), img_mark)

    # x_train, y_train
    # PCA降维
    x_train = PCA(n_components=20).fit_transform(x_train)
    # 构建 visual bag of words dictionary
    bow_trainer1 = cv2.BOWKMeansTrainer(50)
    bow_trainer1.add(np.float32(x_train))
    v1 = bow_trainer1.cluster()
    x_train = np.hstack((x_train, np.dot(x_train, v1.T)))
    x_train, y_train = np.array(x_train), np.array(y_train)

    # x_test, y_test
    # PCA降维
    x_test = PCA(n_components=20).fit_transform(x_test)
    # 构建 visual bag of words dictionary
    bow_trainer2 = cv2.BOWKMeansTrainer(50)
    bow_trainer2.add(np.float32(x_test))
    v2 = bow_trainer2.cluster()
    x_test = np.hstack((x_test, np.dot(x_test, v2.T)))
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test


def test(x_train, y_train, x_test, y_test):
    # 随机森林，限制最大深度5
    rfc = RandomForestClassifier(max_depth=5)
    rfc.fit(x_train, y_train)
    y_train_predict = rfc.predict(x_train)
    y_test_predict = rfc.predict(x_test)
    print("训练集上的准确率：", accuracy_score(y_train, y_train_predict))
    print("测试集上的准确率：", accuracy_score(y_test, y_test_predict))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_generate()
    test(x_train, y_train, x_test, y_test)
