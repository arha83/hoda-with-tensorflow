from HodaDatasetReader.HodaDatasetReader import read_hoda_dataset
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, applications
import cv2 as cv
import numpy as np


'''
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

classes= [0,1,2,3,4,5,6,7,8,9]

# loading dataset:
print('### loading dataset...')
trainImages, trainLabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Train 60000.cdb', images_height=32, images_width=32, one_hot=False, reshape=True)
testImages, testlabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Test 20000.cdb', images_height=32, images_width=32, one_hot=True, reshape=False)
remainImages, remainLabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/RemainingSamples.cdb', images_height=32, images_width=32, one_hot=True, reshape=True)

# normalizing dataset:
trainImages= trainImages.reshape(trainImages.shape[0], 32, 32, 1)
testImages= testImages.reshape(testImages.shape[0], 32, 32, 1)
remainImages= remainImages.reshape(remainImages.shape[0], 32, 32, 1)
trainImages= cv.cvtColor(trainImages, cv.COLOR_GRAY2RGB) / 255.0
testImages= cv.cvtColor(testImages, cv.COLOR_GRAY2RGB) / 255.0
remainImages= cv.cvtColor(remainImages, cv.COLOR_GRAY2RGB) / 255.0

# building the model:
print('### building the model...')
baseModel= applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights='imagenet')
baseModel.trainable= False
averageLayer= layers.GlobalAveragePooling2D()
predictionLayer= layers.Dense(10)
model= models.Sequential([baseModel, averageLayer, predictionLayer])

print(model.summary())










