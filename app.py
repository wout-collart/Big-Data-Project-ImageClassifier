# app.py

import os
from flask import Flask, request, make_response, jsonify, render_template, flash, send_from_directory, redirect
from werkzeug.utils import secure_filename, redirect
from fastai.vision.all import *

from fastai.data.external import *

# codeblock below is needed for Windows path #############
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
learner = load_learner('vehicle_model.pkl')
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def predict():
    if 'image' not in request.files:
        #return {'error': 'no image found, in request.'}, 400
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    request.files['image'].save("./uploads/" + file.filename)
    if file.filename == '':
        #return {'error': 'no image found. Empty'}, 400
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img = PILImage.create(file)
        pred = learner.predict(img)
        print(pred)
        #return {'success': pred[0]}, 200
        flash(pred[0])

        return render_template("index.html", image_file_name=file.filename)

    return {'error': 'something went wrong.'}, 500


@app.route("/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    #port = os.getenv('PORT', 5000)
    #app.run(debug=True, host='0.0.0.0', port=port)
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run()