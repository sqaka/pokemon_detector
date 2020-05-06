from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
 
import argparse
import cv2
from PIL import Image


# 再設計中 ========
HUMAN_NAMES = {
  0: u"XP　を　あおっていた　ポケモン",
  1: u"NOAH　を　あおっていた　ポケモン",
  2: u"Wowbit　を　あおっていた　ポケモン",
  3: u"どうやら　ポケモン　では　ない",
}


def evaluation(img_path, model_path):

# ==================

# 顔を検出して画像を切り取る
def faceDetectionFromPath(path, size):
    # print(f"path: {path}")
    cvImg = cv2.imread(path)
    # print(f"cvImg.shape: {cvImg.shape}")
    cascade_path = "./lib/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    # print(f"cascade.empty(): {cascade.empty()}")
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    faceData = []
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, fx=float(size/faceImg.shape[0]),fy=float(size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg) 
    return faceData
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./mymodel.h5')
    parser.add_argument('--testpath', '-t', default='./testimage5.png')
    args = parser.parse_args()
    # print("agrs: ", args)
    
    # ポケモン3種とそれ以外の計4ラベルに分ける
    num_classes = 4
    img_rows, img_cols = 128, 128
 
    ident = [""] * num_classes
    # label.txt
    for line in open("label.txt", "r"):
        dirname = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dirname
 
    model = load_model(args.model)
    faceImgs = faceDetectionFromPath(args.testpath, img_rows)
    imgarray = []
    for faceImg in faceImgs:
        faceImg.show()
        imgarray.append(img_to_array(faceImg))
    imgarray = np.array(imgarray) / 255.0
    imgarray.astype('float32')
 
    preds = model.predict(imgarray, batch_size=imgarray.shape[0])
    for pred in preds:
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                print("this is {}".format(ident[pre_i]))
 
if __name__ == '__main__':
    main()