import os

import flask
import numpy as np
from flask import request
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

from extract_bottleneck_features import *

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
ResNet50_model_base = ResNet50(weights="imagenet")

def load_model2():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global ResNet50_model_base
    ResNet50_model_base = ResNet50(weights="imagenet")

def load_model():
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.load_weights('../application_data/weights.best.Resnet50.hdf5')
    global model
    model = Resnet50_model

def dognames():
    import json
    with open('../application_data/dog_names.json', 'r') as f:
        dog_names = json.load(f)
    return dog_names

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):

    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_base.predict(img))

def Resnet50_predict_breed(img_path, Resnet50_model=model):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    idx = np.argmax(predicted_vector)
    dog_names = dognames()
    name = dog_names[idx]
    name = name.split('.')[1]
    probal = str(round((predicted_vector.flatten())[idx], 2))
    return name, probal

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/", methods=['GET', 'POST'])
def predict():
    data = {"success": False}
    if request.method == 'POST':
        # ensure an image was properly uploaded to our endpoint
        #print(flask.request)
        if flask.request.method == "POST":
            print("fase 1")
            if flask.request.files.get("file"):
                file = request.files['file']
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                #file=flask.request.files.get("file")
                print(file)
                print(filepath)
                label, prob = Resnet50_predict_breed(img_path, Resnet50_model=model)
                print(label)
                print(prob)

                r = {"Breed": label, "Probability": prob}
                data["Dog"] = []
                data["Dog"].append(r)

                # indicate that the request was a success
                data["success"] = True

        # return the data dictionary as a JSON response
        return flask.jsonify(data)
    return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
              <p><input type=file name=file>
                 <input type=submit value=Upload>
            </form>
            '''

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    img_path = '../images/Welsh_springer_spaniel_08203.jpg'
    #print(Resnet50_predict_breed(img_path, Resnet50_model=model))
    print(ResNet50_predict_labels(img_path))
    app.run()
