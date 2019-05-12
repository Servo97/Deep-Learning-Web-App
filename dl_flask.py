from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import base64
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import numpy as np
import skimage
from skimage import data
from matplotlib import pyplot as plt

app = Flask(__name__)

ap = argparse.ArgumentParser()
'''ap.add_argument("-m","--model", required = True,
                help="path to trained model")
ap.add_argument("-l","--labelbin",required = True,
                help="path to label binarizer")
ap.add_argument("-i","--image",required=True,
                help="path to input image")
args=vars(ap.parse_args())
'''
#load the model
def get_model():
    global model,lb
    model = load_model("vehicle0.model")
    model._make_predict_function()
    lb= pickle.loads(open("lb1.pickle","rb").read())
    print('Model Loaded')

def preprocess_image(image, size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size)
    image = np.array(image)
    image = image.astype("float")/255.0
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    
    return image

get_model()
global graph
graph = tf.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,size=(100,100))
    
    with graph.as_default():
        prob = model.predict(processed_image)[0]
    idx = np.argmax(prob)
    label = lb.classes_[idx]
    response = {
        "prediction":{
            "vehicle": label,
            "confidence": prob[idx]*100
        }
    }
    return jsonify(response)


