# -*- coding: utf-8 -*-

import multiprocessing as mp
# import numpy as np
import os
# import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for
# from werkzeug import secure_filename

# from tools import eval

# インスタンス化
app = Flask(__name__)
app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images/default/'

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
        # eval.pyへアップロードされた画像を渡す
        result = eval.evaluation(img_path, './mymodel.h5')
    else:
        result = []
    return render_template('index.html', result=result)
  else:
    # エラーの際の挙動
    return render_template('error.html')

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)