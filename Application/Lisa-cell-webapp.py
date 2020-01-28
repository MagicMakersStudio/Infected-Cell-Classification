
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

# machine learning
import keras
import numpy as np
from keras.models import load_model, Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import h5py
# manipulation de données
from PIL import Image
from glob import glob
from tqdm import tqdm
# webapp
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

#-=-=-=-=-=-=-=-=-=-=-#
#    CRÉER UNE APP    #
#-=-=-=-=-=-=-=-=-=-=-#

app = Flask(__name__)
# on définit le dossier où les images seront enregistrées
UPLOAD_FOLDER = './imageenregistre'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#-=-=-=-=-=-=-=-=-=-=-#
#   CHARGER MODÈLE    #
#-=-=-=-=-=-=-=-=-=-=-#

K.clear_session()
model = load_model("model.h5")
graph = tf.get_default_graph()
#-=-=-=-=-=-=-=-=-=-=-#
#  PAGE POUR UPLOAD   #
#-=-=-=-=-=-=-=-=-=-=-#

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#    PAGE ACCUEIL - UPLOAD IMAGE    #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('index.html')

#-=-=-=-=-=-=-=-=-=-=-=-=-#
# PREDICTION SUR L'IMAGE  #
#-=-=-=-=-=-=-=-=-=-=-=-=-#

@app.route('/analyse')
def anlyse():
    file = glob("imageenregistre/*.png")
    global graph
    with graph.as_default():
        for image in tqdm(file):
            imgi = Image.open(image).convert("L").resize((100, 100))
            imgi = np.array(imgi)
            imgi = imgi.reshape(100, 100,1)
            imgi = imgi.astype('float32')
            imgi /= 255
            imgi = np.expand_dims(imgi, 0)
            predicty = model.predict(imgi)
            predicty = np.argmax(predicty)

            if predicty == 0 :
                lien = "infected"
            if predicty == 1 :
                lien = "uninfected"

    return redirect(url_for(lien))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#   PAGE PREDICTION INFECTED  #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

@app.route('/infected/')
def infected():
	return render_template('infected.html')

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# PAGE PREDICTION UNINFECTED  #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

@app.route('/uninfected/')
def uninfected():
	return render_template('uninfected.html')
