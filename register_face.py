#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/10
# project = register_face
import os.path
import time

import cv2
import dlib
import numpy as np


# save face pic
class Face_Register:
    def __init__(self):
        self.path_photo = "data/data_from_photo/"
        self.font = cv2.FONT_ITALIC

    # Mkdir for saving photos
    def pre_work_mkdir(self):
        if not os.path.isdir(self.path_photo):
            os.mkdir(self.path_photo)

    def get_photo_data(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.pre_work_mkdir()
        detector = dlib.get_frontal_face_detector()
        save_flag = False
        self.pre_work_mkdir()
        while True:
            ret, frame = cap.read()  # 返回每一帧，ret=True prove it's ok , Variable frame means the image of each frame
            input_key = cv2.waitKey(1)
            if input_key == ord('q'):
                break
            faces = detector(frame, 0)
            for i, face in enumerate(faces):
                # 6. 判断人脸矩形框是否超出 480x640 / If the size of ROI > 480x640
                height = face.bottom() - face.top()
                weight = face.right() - face.left()
                hh = (face.bottom() - face.top()) // 2
                ww = (face.right() - face.left()) // 2
                if face.right() + ww > 640 or face.left() - ww < 0 or face.bottom() + hh > 480 or face.top() - hh < 0:
                    cv2.putText(frame, "out of range", (20, 30), self.font, 0.8, (0, 0, 255))
                    color_rectangle = (255, 0, 0)
                else:
                    color_rectangle = (255, 255, 255)
                cv2.rectangle(frame, (int(face.left() - ww), int(face.top() - hh)),
                              (int(face.right() + ww), int(face.bottom() + hh)), color_rectangle, 2)
                if input_key == ord('s'):
                    save_flag = True
                if save_flag:
                    save_flag = False
                    name=str(input("请输入人名\n"))
                    # 检查有没有创建目录,没有就建立
                    img_blank = np.zeros((height * 2, weight * 2, 3), np.uint8)
                    for ii in range(height * 2):
                        for jj in range(weight * 2):
                            img_blank[ii][jj] = frame[face.top() - hh + ii][face.left() - ww + jj]
                    cv2.imwrite(self.path_photo + '{}.jpg'.format(name), img_blank)

                cv2.imshow('new frame', frame)

        cap.release()
        cv2.destroyAllWindows()


face_register = Face_Register()
face_register.get_photo_data()
