import numpy as np
from HodaDatasetReader.HodaDatasetReader import read_hoda_dataset
import os
from tensorflow.keras import datasets, layers, models, losses, applications, optimizers
import cv2 as cv
import tensorflow as tf

'''
---uncomment for using gpu
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

classes= [0,1,2,3,4,5,6,7,8,9]

# loading dataset:
print('### loading dataset...')
trainImages, trainLabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Train 60000.cdb', images_height=32, images_width=32, one_hot=False, reshape=True)
testImages, testLabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Test 20000.cdb', images_height=32, images_width=32, one_hot=False, reshape=True)
remainImages, remainLabels= read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/RemainingSamples.cdb', images_height=32, images_width=32, one_hot=False, reshape=True)

# normalizing dataset:
print('### normalizing dataset...')
trainImages= trainImages.reshape(trainImages.shape[0], 32, 32, 1)
testImages= testImages.reshape(testImages.shape[0], 32, 32, 1)
remainImages= remainImages.reshape(remainImages.shape[0], 32, 32, 1)


# building the model:
print('### building the model...')
model= models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
# training:
model= models.load_model('./myModel/myMNV2.h5')
'''print('### training...')
baseLearningRate= 0.0001
model.compile(
    optimizer= optimizers.RMSprop(baseLearningRate),
    loss= losses.SparseCategoricalCrossentropy(from_logits= True),
    metrics=['accuracy'])
model.fit(
    trainImages, trainLabels,
    epochs=2,
    validation_data=(testImages, testLabels))
model.save('./myModel/myMNV2.h5')'''

# predicting:
os.system('cls')
print('### predicting...')
print('image should be in \'test images\' folder.')
print('image format should have \'.png\' format.')
print('enter \'q\' for exit.\n')
while True:
    imageName= input('please enter image name: ')
    os.system('cls')
    if imageName == 'q': break
    image= cv.imread('./test images/'+imageName+'.png')
    image= cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    pre= np.expand_dims(image, 0)
    predictions= model.predict([pre])
    maxi= np.where(predictions[0] == np.amax(predictions[0]))[0][0]
    for i in range(10): print(classes[i], ': ', predictions[0][i], sep='')
    print(f'closest class to {imageName}: {classes[maxi]}')








