#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/20
# project = face_reco_from_camera
import logging
import os.path

import cv2
import dlib

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
import numpy as np
import pandas as pd

detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


class Face_Recognizer():
    def __init__(self):
        self.face_feature_known_list = []  # 存储数据库中人名的特征点
        self.face_name_known_list = []  # 存储数据库中的人名

        self.frame_cnt = 0
        self.path_face_database = 'data/features_all.csv'

        self.font = cv2.FONT_ITALIC

    def get_face_database(self):
        if os.path.exists(self.path_face_database):
            csv_rd = pd.read_csv(self.path_face_database, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])  # 存储人名
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == "":
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return True
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return False

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        distance = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return distance

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "face recognize", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame: " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0))

    def draw_name(self, img_rd, names):
        for i,name in enumerate(names):
            cv2.putText(img_rd, "name: " + name, (i*20, 200), self.font, 0.8, (0, 255, 0))

    def process(self, cap):
        if self.get_face_database():
            while cap.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame %d starts", self.frame_cnt)  # 查看第几帧
                ret, img_rd = cap.read()
                faces = detector(img_rd, 1)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                else:
                    # 目前只考虑了一张脸的情况
                    self.draw_note(img_rd)
                    current_frame_face_feature_list = []
                    current_face = ['unknown']*len(faces)
                    if len(faces) != 0:
                        for i in range(len(faces)):  # 保存所有每一帧所有面部特征
                            shape = predictor(img_rd, faces[i])
                            current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                        current_frame_face_distance = []
                        for k in range(len(faces)):
                            temp = []
                            for j in range(len(self.face_feature_known_list)):
                                temp.append(self.return_euclidean_distance(current_frame_face_feature_list[k],
                                                                           self.face_feature_known_list[j]))
                            current_frame_face_distance.append(temp)
                        current_frame_face_distance=np.array(current_frame_face_distance)
                        current_frame_face_distance_MinVal=current_frame_face_distance.min(axis=1)
                        current_frame_face_distance_Min_index=current_frame_face_distance.argmin(axis=1)
                        print(current_frame_face_distance_MinVal)
                        print(current_frame_face_distance_Min_index)
                        for i,val in enumerate(current_frame_face_distance_MinVal):
                            if val <0.35:
                                current_face[i]=self.face_name_known_list[current_frame_face_distance_Min_index[i]]
                        # 矩形框 / Draw rectangle
                        for kk, d in enumerate(faces):
                            # 绘制矩形框
                            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                          (255, 255, 255), 2)
                    if current_face:
                        self.draw_name(img_rd, current_face)  # 先测一个人
                    cv2.imshow("camera", img_rd)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 480)  # 3 代表宽度width， 480表示高度的数值为480
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
