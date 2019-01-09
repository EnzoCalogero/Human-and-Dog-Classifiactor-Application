import argparse
import sys

import libs.classifier_lib as cl_lib


def output(img_path):
    if cl_lib.dog_detector(img_path):
        print(cl_lib.dog(img_path, Resnet50_model))
        cl_lib.img_show(img_path)

    elif cl_lib.face_detector(img_path):
        print(cl_lib.human(img_path, Resnet50_model))
        cl_lib.img_show(img_path)
    else:
        print(cl_lib.not_found())


def check_arg(args=None):
    parser = argparse.ArgumentParser(description=
                                     'Application to identify whether an image is a dog or a Human '
                                     'and then give the prediction of the dog breed')
    parser.add_argument('-th', '--testHuman',
                        help='Test the application with a Human image',
                        action="store_true",
                        required=False)

    parser.add_argument('-td', '--testDog',
                        help='Test the application with a Dog image',
                        action="store_true",
                        required=False)

    parser.add_argument('-f', '--file',
                        help='Give the image to be classified using the filepath/file name')

    results = parser.parse_args(args)
    return (results.testHuman,
            results.testDog,
            results.file)

if __name__ == '__main__':
    th, td, f = check_arg(sys.argv[1:])
    Resnet50_model = cl_lib.model()

    if th:
        output('../images/download.png')
    if td:
        output('../images/Welsh_springer_spaniel_08203.jpg')
    if f:
        output(f)
