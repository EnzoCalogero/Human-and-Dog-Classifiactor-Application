import os

import cv2
import flask
import numpy as np
from flask import request, render_template
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

from extract_bottleneck_features import *

# Initialize our Flask application
app = flask.Flask(__name__)
# Defining the Uploads folder
app.config['UPLOAD_FOLDER'] = 'static'


model = None

#  load the pre-trained Keras model (here we are using a model
#   pre-trained on ImageNet and provided by Keras
ResNet50_model_base = ResNet50(weights="imagenet")


def load_model():
    """
        load the Keras model created on jupiter
        :return: None
    """
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.load_weights('../application_data/weights.best.Resnet50.hdf5')
    global model
    model = Resnet50_model


def dognames():
    """
    Populate the dictionary "dog_names", where for each dog's breed is reported the associated code.
    :return: dictionary of the breeds
    """
    import json
    with open('../application_data/dog_names.json', 'r') as f:
        dog_names = json.load(f)
    return dog_names


def path_to_tensor(img_path):
    """
    Trasform an image into a 4D tensor

    :param img_path: image
    :return: tensor in 4D with shape (1, 224, 224, 3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):
    """
    Returns prediction vector for image located at img_path
    :param img_path: image file
    :return: dog breed int
    """
    #
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_base.predict(img))


def Resnet50_predict_breed(img_path, Resnet50_model=model):
    """
    return the dog breed and its probability
    :param img_path: image file
    :param Resnet50_model:
    :return: (dog breed name, associate probability), (string, float)
    """
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    idx = np.argmax(predicted_vector)
    dog_names = dognames()
    name = dog_names[idx]
    name = name.split('.')[1]
    probal = str(round((predicted_vector.flatten())[idx], 2))
    return name, probal


def prepare_image(image, target):
    """
    Standardasize the image
    :param image:
    :param target: dimension
    :return: image
    """
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


def dog_detector(img_path):
    """
    look for any dog breeds, and report if identify one
    :param img_path:
    :return: boolean, true if identify a dog breed.
    """
    prediction = ResNet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)


def face_detector(img_path):
    """
    Look for a face in the picture
    :param img_path:
    :return: boolean, true if a face is identify
    """
    face_cascade = cv2.CascadeClassifier('../application_data/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


@app.route("/", methods=['GET', 'POST'])
def predict():
    data = {"title": False}
    if request.method == 'POST':
        if flask.request.method == "POST":
            if flask.request.files.get("file"):
                # Up-load the image file
                file = request.files['file']
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                data["filepath"] = filepath

                # is recognised as a Human?
                is_human = face_detector(filepath)
                data['is_human'] = is_human

                # is recognised  as a Dog?
                is_dog = dog_detector(filepath)
                data["is_dog"] = is_dog

                # identify the most similar dog breed for the image.
                label, prob = Resnet50_predict_breed(filepath, Resnet50_model=model)

                # Populate the labels for the UI
                if is_human & is_dog:
                    data["title"] = "Human or Dog!!!"
                    data["breed"] = "You are {}!".format(label)
                elif is_human:
                    data["title"] = "Human!!!"
                    data["breed"] = "You look like a ... {}!".format(label)
                elif is_dog:
                    data["title"] = "Dog!!!"
                    data["breed"] = "You are {}!".format(label)
                else:
                    data["title"] = "You Do not look like a Human or a Dog!!!"
                    data["breed"] = "what ever you are you look like a .... {}!".format(label)

                data["probability"] = "With a probability of {}%!".format(prob)
        # return case for "post"
        return render_template('results.html', data=data)
    # return case for "get"
    return render_template('start.html')

if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...\n"
          "please wait until server has fully started\n"
          "then connect to the URL http://127.0.0.1:5000")
    load_model()
    app.run()
