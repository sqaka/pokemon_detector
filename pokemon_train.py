# python2系を3系にするためのものなので、必要ない
# from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras import backend as K
import argparse
from load_images import load_images_from_labelFolder
 
# 最初に定数を指定する
batch_size = 20　# 一度に学習するデータサイズ
num_classes = 3　# 分類するラベル数
epoch = 30　# 全データを何回学習するか

# これなに？
img_rows, img_cols = 128, 128
 
# これなに？
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', default='.\\images')
args = parser.parse_args()　
 
# データ読み込み testデータとtrainデータに分割する
(x_train, y_train), (x_test, y_test) = load_images_from_labelFolder(args.path,img_cols, img_rows, train_test_ratio=(6,1))

# これなに？
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# 画像データは0～255の値をとるので、255で割ることでデータを標準化する
# .astype('float32')でデータ型を変換する
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test /= 255

# データの数を出力して確認
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# ラベルデータをone-hot-vector化し、0と1だけでどのラベルなのか表せるようにする
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Sequentialクラスをインスタンス化
# https://keras.io/ja/getting-started/sequential-model-guide/
model = Sequential()

# 中間層
# Conv2D 3×3のフィルタを各マスにかけ、16枚の出力データを得られるように指定している
# kernel_size(フィルタ)の部分は省略可能　(Conv2D(32,(3,3))のように
model.add(Conv2D(32, kernel_size=(3,3),
            # relu（Rectified Linear Unit）は、特徴を際立てるための活性化関数
            # https://arakan-pgm-ai.hatenablog.com/entry/2018/11/07/090000
            activation='relu',
            # 一番最初の層にだけinput_shapeを指定する必要がある
            input_shape=input_shape))
# MaxPooling2Dで出力をダウンスケールする。(2, 2)は画像をそれぞれの次元で半分にする
# https://keras.io/ja/layers/pooling/
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# Dropoutを設定し、過学習を防止する
# https://qiita.com/shu_marubo/items/70b20c3a6c172aaeb8de
model.add(Dropout(0.2))

# ZeroPadding2Dは、画像のような2次元入力のためのレイヤーで、画像テンソルの上下左右にゼロの行と列を追加する
# デフォルトが(1, 1)なので一旦これでよい https://keras.io/ja/layers/convolutional/
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(96, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))
 
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(96, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
 
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=num_classes, activation='softmax'))
 
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
 
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epoch,
            verbose=1,
            validation_data=(x_test, y_test))
 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# モデルの概要を出力する。見て確認するためのもので、処理に影響はない
model.summary()

# 学習済みモデルを保存
model.save('mymodel.h5')