from google.colab import drive
drive.mount("/content/drive")

 
import os

os.environ['KAGGLE_USERNAME'] = "smeerdadh" 

os.environ['KAGGLE_KEY'] = "6466405f4dff3392a557c1846f3df071"



#!kaggle datasets download -d arashnic/hr-analytics-job-change-of-data-scientists

#!kaggle datasets download -d manideep1108/tusimple

!kaggle datasets download -d thomasfermi/lane-detection-for-carla-driving-simulator
!unzip -q "/content/lane-detection-for-carla-driving-simulator.zip"



#import zipfile
#with zipfile.ZipFile("/content/tusimple.zip", 'r') as zip_ref:
#    zip_ref.extractall("/content/drive/MyDrive/project")

!pip install fastseg
!pip install fastai --upgrade
from fastai.vision.all import *
import os
import matplotlib.pyplot as plt
import cv2
import albumentations  as albu
albu.__version__
