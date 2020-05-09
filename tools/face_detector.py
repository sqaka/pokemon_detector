from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

import cv2
from PIL import Image

# 顔を検出して画像を切り取る
def faceDetectionFromPath(img_path, model_path):
    # print(f"path: {path}")
    cvImg = cv2.imread(img_path)
    print(f"cvImg.shape: {cvImg.shape}")
    cascade_path = "./lib/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    # print(f"cascade.empty(): {cascade.empty()}")
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, 
                                        minNeighbors=1, minSize=(1, 1))
    faceData = []
    size = 128
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, 
                             fx=float(size/faceImg.shape[0]),
                             fy=float(size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg) 
    return faceData