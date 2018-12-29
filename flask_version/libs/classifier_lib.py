import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from tqdm import tqdm

from libs.extract_bottleneck_features import *


def dog(img_path, Resnet50_model):
    name, prob = Resnet50_predict_breed(img_path, Resnet50_model)
    return "Hello, dog!\nThe dog looks like a ...{}  (Probability: {})".format(name, prob)


def human(img_path, Resnet50_model):
    name, prob = Resnet50_predict_breed(img_path, Resnet50_model)
    return "Hello, human!\n You look like a ...{}  (Probability: {})".format(name, prob)


def not_found():
    return "Sorry, you look like neither dog or human..."


def img_show(img_path):
    cv_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()


def dognames():
    import json
    with open('../application_data/dog_names.json', 'r') as f:
        dog_names = json.load(f)
    return dog_names


def model():
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Resnet50_model.add(Dense(133, activation='softmax'))

    Resnet50_model.load_weights('../application_data/weights.best.Resnet50.hdf5')
    return Resnet50_model


def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('../application_data/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


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


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def Resnet50_predict_breed(img_path, Resnet50_model):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    idx = np.argmax(predicted_vector)
    dog_names = dognames()
    name = dog_names[idx]
    name = name.split('.')[1]
    probal = str(round((predicted_vector.flatten())[idx], 2))
    return name, probal
