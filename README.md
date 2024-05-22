# Verification signature using Neural Network
This program was created to verificate offline signatures with usage of two methods using neural network. Those methods are convolutional neural network and siamese convolutional neural network. This programme was created for bachelor thesis at BUT FEEC as a school work.
## Instalation
Program is installed by downloading the code (or cloning) and installing requirements.txt. Create data and test folder and load in to these folders datasets. Some of datasets were structuralized for better work. For additional info feel free to contact me at 241015@vutbr.cz. Then run the main.py file.  
## Files in program
### conf.py
Contains configuration files, such as names of datasets or directory path to models. Change `MODEL_DIR` path to path where you have saved your models.
### data
Create this folder for data of training datasets. 
### test
Create this folder for data of testing datasets
### main.py
Running main.py set up everything, starts the program and let user choose what should program do (train, evaluate model or predict genuinity of image) and set the parameters.
### model.py
Contains model architecture.
### loader.py
Includes dataloaders, and augmentation functions. Also functions such as `convert_to_image` which are usefull for debuging and for python in console predictions
### functions.py
Contains supportive functions for models and loader such as callbacks, additional feature extractions, images plotting and CNN interpretation functions.
### validator.py
Contains functions for testing trained models and for predicting genuinity of signatures. Also contains functions for interpreting CNN models and testing packages of models.
