import tensorflow as tf
from keras.layers import Input, Lambda,Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


IMAGE_SIZE = [244,244]
train_path = "D:\\test\\Emotions Dataset\\train"
test_path = "D:\\test\\Emotions Dataset\\test"

vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

print(f"model vgg is being used")

for layer in vgg.layers:
    layer.trainable=False

folders=glob("D:\\test\\Emotions Dataset\\train\\*")

x=Flatten()(vgg.output)

print("flattening layer added")

prediction=Dense(len(folders),activation='softmax')(x)

print("Last layer added on top of pre-trained vggnet16 model")

model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data augmentation : 
train_data=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_data=ImageDataGenerator(
    rescale=1./255)

training_data=train_data.flow_from_directory('D:\\test\\Emotions Dataset\\train',
                              target_size=(244,244),
                               batch_size=32,
                                class_mode='categorical' )

testing_data=test_data.flow_from_directory('D:\\test\\Emotions Dataset\\test',
                              target_size=(244,244),
                               batch_size=32,
                                class_mode='categorical' )

print("Data augmentation completed")

r=model.fit(
    training_data,
    validation_data = testing_data,
    epochs=5,
    steps_per_epoch = len(training_data),
    validation_steps = len(testing_data)
)

print("model trained")

print("Plotting the accuracy and loss")
# plotting(

plt.plot(r.history['loss'],label='train_loss')
plt.plot(r.history['val_loss'],label='val_losss')
plt.legend()
plt.show()
plt.savefig('Val_Loss')

#saving our model
print("saving the model")
from keras.models import load_model
model.save('facefeatures_new_model.h5')

