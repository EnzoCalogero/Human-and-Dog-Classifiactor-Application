###  Project Overview

  

 This project aim to perform two distinct classifications:

1.  given an image, it will identify whether is a human or a dog.  
2.  from the image it will identify for dog which breed it  and for human which breed he or she looks like.

####    Motivation

  In this project,  two classifiers have been created, tested and trained using the jupiter notebook.

Then, the model has been implemented for both a web app (Flask) and for an argparse CLI application, where a given image is provided and the algorithm will first identify whether dog or human. Then it will identify canine’s breed.  



##  **Instructions**

The project is composed of 3 independent components (Jupiter notebook, Argversion  app, flask app).

The Jupiter notebook is where the pipeline have been created and tested.


Then the final  algorithms have been implemented in two versions:  the argversion and the flask versions.


### **Instructions**

  **To replicate the Jupiter Notebook**

1. Clone the repository and navigate to the downloaded folder.
2. ```bash
   git clone https://github.com/EnzoCalogero/Human-and-Dog-Classification-Application.git
   cd  Human-and-Dog-Classification-Application
   ```

3. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 	Unzip the folder and place it in the repo, at 	location `path/to/dog-project/dogImages`. 	The `dogImages/` folder 	should contain 133 folders, each corresponding to a different dog breed.
4. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). 	Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`. If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.
5. Install the libraries listed on the paragraph below (Required Libraries).
6. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.

```bash
jupyter notebook dog_app.ipynb
```

(please note,  the notebook was implemented using a GPU ).



**To replicate the Flask_version**

1. Clone the repository and navigate to the downloaded folder.

2. ```bash
     git clone https://github.com/EnzoCalogero/Human-and-Dog-Classification-Application.git
       cd  Human-and-Dog-Classification-Application/flask_version
     ```
  ```bash

3. then run the following command:
​```bash
python new_app.py
  ```
4. Go to the browser at the URL ```127.0.0.1:5000```

5. Upload an image by selecting one image file and click on the button labelled upload.

   ![first page flask](/images/flask_first_b.png)

6. The result page will display whether is a dog or a human and which dog breed is most similar with an associated probability.

   ![result Page Flask]("/images/flask_second.png")

**To replicate the argparse version...**

1. Clone the repository and navigate to the downloaded folder.
2. ```bash
   git clone https://github.com/EnzoCalogero/Human-and-Dog-Classification-Application.git
   cd  Human-and-Dog-Classification-Application/argparse_version
   ```

3. To Run the script we have 3 different parameters:
    for a given file like the example below:
```bash
	python  Human_dog_classifer.py -f ‘../images/Welsh_springer_spaniel_08203.jpg’
```

to test the application for Human 
```bash
	python  Human_dog_classifer.py -th
```
to test the application for dog 
```bash
	python  Human_dog_classifer.py -td
```

### **Files**

**dog_app.ipynb**: jupiter notebook, used to create, test and train the models used on the (flask and argb version) applications.

**extract_bottleneck_features.py**: bottleneck features of a pre-trained network for VGG16, VGG19, Resnet50, Xception, and IneptionV3.

**application_data/**:  data folder used by the two versions of the application.

**application_data/dog_names.json**: json file contains the dictionary of the dog breeds with the associated numeric code

​		(used for deep learning classifier)  

**application_data/haarcascade_frontalface_alt.xml**: OpenCV pre- trained model to detect human face in images.

**application_data/weights.best.Resnet50.hdf5**:  File of the weights used by the Keras neural network to identify the  dogs breeds.

**argparse_version**/ folder contains the CLI version of the application.

**argparse_version/Human_dog_classifer.py**: the script to run the argparse version of the application.

***arggarse_version/libs/classifer_lib.py***: all the functions required by the script  Human_dog_classifer.py.

***flask_version/***: folder contains the flask version of the application.

***flask_version/new_app.py***: the script to run the flask version of the application.

***flask_version/templates***: folder where the flask templates are located.

***flask_version/templates/start.html***: flask templates for the first page of the application.

***flask_version/templates/results.html***: flask templates for the result page of the application.

***flask/Uploads***: where the application store the upload image file.

***images/***: where all repository images are located.



### Required libraries
(based on the anaconda 3.6.3)

. Flask 1.2.2

. numpy 1.14.5

. pandas 0.23.4

. scikit-learn 0.20.0

. keras 2.20

. tensorflow 1.9.0

. opencv-python 3.4.5.20

. matplot 2.2.2

. pillow 4.2.1
