#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/19
# project = features_extraction_to_csv
import csv
import logging
import os
import re

import cv2
import dlib
import numpy as np

photo_path_dir = "data/data_from_photo/"
# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')


def return_128d_features(img_path):
    img = cv2.imread(img_path)
    faces = detector(img, 1)
    logging.info("正在检测人脸图片 {} ".format(img_path))
    if len(faces) != 0:
        shape = predictor(img, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img, shape)  # 128个特征点
    else:
        face_descriptor = 0
        logging.warning("检测不到人脸")
    return face_descriptor


def return_features_mean_personX(img_path_dir):
    features_list_personX = []
    if img_path_dir:
        logging.info("正在读人脸图片 {} ".format(img_path_dir))
        features_128d = return_128d_features(photo_path_dir + img_path_dir)
        # 检测到人脸
        if features_128d != 0:
            features_list_personX.append(features_128d)
    else:
        logging.warning("文件夹 {} 内图像为空".format(img_path_dir))
    if features_list_personX:
        # 取均值应该是一个人拍了很多照片，取一个人特征点的均值的均值
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX


def main():
    logging.basicConfig(filename='log.log', level=logging.INFO)
    with open('data/features_all.csv', 'w+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        logging.info("person:{}".format(photo_path_dir))
        for i,path in enumerate(os.listdir(photo_path_dir)):
        # Get the mean/average features of face/personX, it will be a list with a length of 128D
            features_mean_personX = return_features_mean_personX(path)
            print(path.split('.')[0])
            features_mean_personX = np.insert(features_mean_personX, 0, "{}".format(path.split('.')[0]), axis=0)
            writer.writerow(features_mean_personX)
    logging.info("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")


if __name__ == '__main__':
    main()
