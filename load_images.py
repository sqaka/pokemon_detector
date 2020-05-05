import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array
import pathlib
# 正規表現モジュールを利用可能にする。
# https://note.nkmk.me/python-re-match-search-findall-etc/
# import re
 
def load_images_from_labelFolder(path, img_width, img_height, train_test_ratio=(9,1)):
    pathsAndLabels = []
    label_i = 0
    # globでフォルダの中身をglobする
    data_list = glob.glob(path + '*')
    print(str(path))
    # ラベル分けのtxtデータを読み込ませる
    datatxt = open('label.txt' ,'w')
    print('data_list', data_list)
    for dataFolderName in data_list:
        pathsAndLabels.append([dataFolderName, label_i])
        # pattern = r".*/(.*)" #pathが階層を色々含んでいるので、画像が入っているフォルダ名だけを取っている。
        # matchOB = re.finditer(pattern, dataFolderName)
        directoryname = pathlib.PurePath(dataFolderName).name
        # if matchOB:
            # for a in matchOB:
                # directoryname += a.groups()[0]
        datatxt.write(directoryname + "," + str(label_i) + "\n") # 誰が何番かテキストで保存しとく
        label_i = label_i + 1
    datatxt.close()
    allData = []
    for pathAndLabel in pathsAndLabels: # 全画像のパスとそのラベルをタプルでリストにし
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "**/*.png")
        for imgName in imagelist:
            allData.append((imgName, label))
    allData = np.random.permutation(allData) # シャッフルする。
 
    train_x = []
    train_y = []
    for (imgpath, label) in allData: #kerasが提供しているpreprocessing.imageでは画像の前処理のメソッドがある。
        img = load_img(imgpath, target_size=(img_width,img_height)) # 画像を読み込む
        imgarry = img_to_array(img) # 画像ファイルを学習のためにarrayに変換する。
        train_x.append(imgarry)
        train_y.append(label)
 
    threshold = (train_test_ratio[0]*len(train_x))//(train_test_ratio[0]+train_test_ratio[1])
    test_x = np.array(train_x[threshold:])
    test_y = np.array(train_y[threshold:])
    train_x = np.array(train_x[:threshold])
    train_y = np.array(train_y[:threshold])
 
    return (train_x, train_y), (test_x, test_y)

if __name__ == '__main__':
    (train_x,train_y),(_,_) = load_images_from_labelFolder('/img_data/', 128, 128)
    print('trainx.shape:',train_x.shape)