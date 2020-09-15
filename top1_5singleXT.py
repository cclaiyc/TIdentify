# coding=utf-8
# top1 top5
import DentNet_ATT_Test   # 1.改网络
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from math import sqrt
import cv2
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#测试集 # 注册集old173
TFRecordFilename = r".\testXT.txt"
TFRecordFilenameXR = r".\testXR.txt"

#
test_modeldir =r".\savefile\testmodels" #需要测试模型的.index文件路径
model_dir=r".\savefile\models" #模型原保存路径
acc_txt = r".\savefile\acc.txt"

counta = 0
count = 0
counta5 = 0
count5 = 0
classnum = 9490
inputsize = 128

def readImg(imageLabelPath,size):
    imgList = []
    labelList = []
    m = 0
    f1 = open(imageLabelPath, "r", encoding="utf-8")
    ImgPaths = f1.readlines()
    for line in ImgPaths:
        ImgPath = line.split(" ")                             # 获取一条图片路径
        img = Image.open(ImgPath[0]).convert('L')
        img = img.resize(size)
        img_im = np.array(img)
        imgList.append(img_im)

        label = ImgPath[1]
        label = label[:-1]
        labelList.append(int(label))
        m+=1
    print("Total Images:", m)
    return imgList,labelList

def concat3Img(TFRecordFilename):
    imgList,labelList = readImg(TFRecordFilename, (inputsize,inputsize))
    picNum = len(imgList)
    img_batch = np.zeros((picNum,inputsize,inputsize,1))        # 输出图片集
    for i in range(0,picNum):
        image = imgList[i]
        image = np.reshape(image, (inputsize, inputsize, 1))
        image = image.astype(np.float32)                        # 把图片的像素值转换为浮点数

        img_batch[i,:,:,:] = image
    return img_batch, labelList, picNum

def cosinDistance(probVectors,gallaryVector,labelXT,labelXR):
    probs = np.array(probVectors)
    probs = probs.T
    gallary = np.array(gallaryVector)
    gallary = gallary.T
    global count  # 正确数
    global counta # 比较数
    global count5  # 正确数
    global counta5 # 比较数
    #print "probs.shape[1]",probs.shape[1]

    for i in range(probs.shape[1]):
        # 每张图片与注册集算相似度
        probstemp = probs[:,i]                   # 提取当前特征(512,1)
        probstempT = probstemp.T                 #(1,512)
        lrs = np.zeros((1,gallary.shape[1]))     # 1x
        lrs2 = np.zeros((1, gallary.shape[1]))
        for j in range(gallary.shape[1]):        # 对于每张注册集特征
            gallarytemp = gallary[:,j]           #(512,1)
            num = np.dot(probstempT,gallarytemp) # 1,1
            gallarytempT = gallarytemp.T
            denum = sqrt(np.dot(probstempT,probstemp)*np.dot(gallarytempT,gallarytemp))
            cos = 1- num/denum
            lrs[0][j] = cos                      # 记录余弦距离 求top1用
            lrs2[0][j] = 1-cos                   # 记录余弦相似度 求top5用

        # top1
        minD = np.argmin(lrs, axis=1)            # 余弦距离最小的注册集的下标
        if labelXR[minD[0]] == labelXT[i]:       # 如果符合标签 判断正确  counta+1
            counta += 1
             # 需要时打印
            # print("top1预测：", labelXR[minD[0]], "真实", labelXT[i])
            """
        else:
            print("top1预测：", labelXR[minD[0]], "真实", labelXT[i])
            """
        count += 1                                      # top1比较次数加1
        """ # 需要时写入文件
        acc_file.write("top1预测："+ str(labelXR[minD[0]]) + "真实" + str(labelXT[i])+"\n")
        """
        # top5
        distance = np.array(lrs2[0])
        minD5 = np.argpartition(distance, -5)[-5:]      # 余弦相似度最大的下标
        for x in range(5):
            if labelXR[minD5[x]] == labelXT[i]:
                counta5 += 1
                break
            """ # 需要时打印
            #else:
                #print("预测top5：", labelXR[minD5[x]], "真实", labelXT[i])
            """
        count5 += 1                                     # top5比较次数加1

    return True


if __name__ == "__main__":

    """读取待检测模型文件名 存在model_list"""
    models_list = []
    for root, dir, files in os.walk(test_modeldir):
        for file in files:
            test_modeldir = os.path.join(root,file)
            test_modelpath = test_modeldir[:-6]
            test_modelpath_1 = test_modelpath.split('\\')[-1]
            models_list.append(model_dir+"\\"+test_modelpath_1)

    """读取全部测试集XT 注册集XR"""
    img_batch, label_batch, picNumXT = concat3Img(TFRecordFilename)
    img_batchXR, label_batchXR, picNumXR = concat3Img(TFRecordFilenameXR)

    image = tf.placeholder(tf.float32, shape=[None, inputsize, inputsize, 1])
    label = tf.placeholder(tf.int64, shape=[None, ])
    """读取特征"""
    model = DentNet_ATT_Test.DentNet_ATT(image, classnum, False)
    label_ = model.feature

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        """对每个模型求top1,5"""
        for nn in range(len(models_list)):
            model_path = str(models_list[nn])
            saver.restore(sess, model_path)

            """
            计算全部注册集特征
            每次只用一张测试集与全部注册集比较 统一计数
            """
            # feaXR = sess.run([label_], feed_dict={image: img_batchXR, label: label_batchXR})
            feaXR = []
            for i in range(picNumXR):
                yXR = np.expand_dims(img_batchXR[i], axis=0)
                yXR_label = np.expand_dims(label_batchXR[i], axis=0)
                feasingleXR = sess.run([label_], feed_dict={image: yXR, label: yXR_label})  # 单个特征
                feaXR.append(feasingleXR)
            feaXR = np.array(feaXR)
            feaXR = np.squeeze(feaXR)

            for i in range(picNumXT):
                y = np.expand_dims(img_batch[i], axis=0)
                y_label = np.expand_dims(label_batch[i], axis=0)
                feaXT = sess.run([label_], feed_dict={image: y, label:y_label})  # 单个特征
                cosinDistance(feaXT, feaXR, y_label, label_batchXR)  # 计算余弦距离

            ## modelName
            acc_file = open(acc_txt, "a", encoding="utf-8")
            acc_file.write(model_path + "\n")
            ## TOP1
            print("true:", counta, "try num:", count)
            acc = float(counta) / count  # 判断正确/比较次数 
            print("Accuracy(Top1):", acc)
            ## TOP5
            acc5 = float(counta5) / count5  # 判断正确/比较次数
            print("Accuracy(Top5):", acc5)
            acc_file.write("Accuracy(Top1)：" + str(acc) + "\n" + "Accuracy(Top5):" + str(acc5) + "\n")
            acc_file.close()

            """下一个模型，重新附初值"""
            counta = 0
            count = 0
            counta5 = 0
            count5 = 0




