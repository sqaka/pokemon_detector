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
# ==================


# 入力画像をモデルに喰わせて結果を算出 
def main():
    # faceData = "./testimage.png"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./tools/model_path')
    parser.add_argument('--testpath', '-t', default=faceData)
    parser.add_argument('--imagepath', '-i', default=img_path)   
    args = parser.parse_args()
    # print("agrs: ", args)
    
    # ポケモン3種とそれ以外の計4ラベルに分ける
    num_classes = 4
    img_rows = 128
    img_cols = 128
 
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

    # モデルを利用し予測する
    # if model_path:
    # 学習後のパラメーターの読み込み
    model = load_model(model_path)
    result = model.predict(imgarray, batch_size=imgarray.shape[0])
  # sess.run(logits)と同じ
   # softmax = logits.eval()
  # 判定結果
   # result = imgarray[0]
  # 判定結果を%にして四捨五入
    rates = [round(n * 100.0, 1) for n in result]
    humans = []
  # ラベル番号、名前、パーセンテージのHashを作成
    for index, rate in enumerate(rates):
        name = HUMAN_NAMES[index]
        humans.append({
          'label': index,
          'name': name,
          'rate': rate
        })
  # パーセンテージの高い順にソート
    rank = sorted(humans, key=lambda x: x['rate'], reverse=True)

  # 判定結果と加工した画像のpathを返す
    return [rank, args.testpath, args.img_path]


    #preds = model.predict(imgarray, batch_size=imgarray.shape[0])
    #for pred in preds:
    #    # 数字を丸める
    #    predRound = np.round(pred)
    #    for pred_i in np.arange(len(predRound)):
    #        if predRound[pred_i] == 1:
    #            predResult = (format(ident[pred_i]))
    #            # 判定結果と加工した画像のpathを返す
    #            return [predResult, 
    #                    args.testpath, 
    #                    args.img_path]
 
# if __name__ == '__main__':
#    main()