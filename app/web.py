# -*- coding: utf-8 -*-
# 再設計中

import multiprocessing as mp
# import numpy as np
import os
# import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from tools.face_detector import faceDetectionFromPath as faceDetect
# from tools.face_detector import facePredict as Pred

UPLOAD_FOLDER = './static/images/default/'

# インスタンス化
app = Flask(__name__)
app.config['DEBUG'] = True

# ルートアクセス時の挙動を設定
@app.route("/")
def index():
    return render_template('index.html')

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
  if request.method == 'POST':
    if not request.files['file'].filename == u'':
        # アップロードされたファイルを保存
        f = request.files['file']
        img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(img_path)
        # face_detector.pyへアップロードされた画像を渡す
        result = faceDetect(img_path, './tools/mymodel.h5')
    else:
        result = []
    return render_template('index.html', result=pred_result)
  else:
    # エラーの際の挙動
    return redirect(url_for('error.html'))

def main():
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
