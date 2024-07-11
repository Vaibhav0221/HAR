from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
import os

app = Flask(__name__)
path_to_model = 'Model2.h5'


V_model = load_model(path_to_model)

def read_image(fn):
    image = Image.open(fn).convert('RGB')
    return np.asarray(image.resize((224, 224)))

def test_predict(test_image):
    result = V_model.predict(np.asarray([read_image(test_image)]))
    itemindex = np.where(result == np.max(result))
    prediction = itemindex[1][0]
    label_map = {
        0: "Sitting",
        1: "Drinking",
        2: "Calling",
        3: "Sleeping",
        4: "Drinking",
        5: "Clapping",
        6: "Dancing",
        7: "Cycling",
        8: "Calling",
        9: "Laughing",
        10: "Eating",
        11: "Fighting",
        12: "Listening_to_music",
        13: "Running",
        14: "Exting"
    }
    prediction = label_map[prediction]
    return prediction

@app.route('/', methods=["GET","POST"])
def home():
    if request.method=="POST":
        image_file = request.files.get('image')
        if image_file:
            prediction = test_predict(image_file)
            return render_template("index.html", result=prediction)
        else:
            return render_template("index.html", result="No image uploaded")
    return render_template("index.html")

