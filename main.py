import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy as sp

import tensorflow as tf
from scipy import ndimage
from shutil import copyfile
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, MaxPool2D, Dense, InputSpec # type: ignore
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

class_name = ["Dogs", "Cats"]

n_dogs = len(os.listdir("/Users/shaswatshrey/Downloads/kagglecatsanddogs_3367a/PetImages/Dog"))
n_cats = len(os.listdir("/Users/shaswatshrey/Downloads/kagglecatsanddogs_3367a/PetImages/Cat"))

n_images = [n_dogs, n_cats]
# fig = px.pie(names=class_name, values=n_images)
# fig.show()
INCLUDE_TEST = True

print(len(os.listdir('tmp/cats-v-dogs/training/cats')))
print(len(os.listdir('tmp/cats-v-dogs/training/dogs')))

print(len(os.listdir('tmp/cats-v-dogs/validation/cats')))
print(len(os.listdir('tmp/cats-v-dogs/validation/dogs')))

print(len(os.listdir('tmp/cats-v-dogs/test/cats')))
print(len(os.listdir('tmp/cats-v-dogs/test/dogs')))

print(tf.config.list_physical_devices('GPU'))

train_gen = ImageDataGenerator(
    rescale=1./255
)
validation_gen = ImageDataGenerator(
    rescale=1./255
)
if INCLUDE_TEST:
    test_gen = ImageDataGenerator(
    rescale=1./255
    )

train_generator = train_gen.flow_from_directory(
    'tmp/cats-v-dogs/training',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary'
)

validation_generator = validation_gen.flow_from_directory(
    'tmp/cats-v-dogs/validation',
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary'
)

if INCLUDE_TEST:
    test_generator = test_gen.flow_from_directory(
        'tmp/cats-v-dogs/test',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

class_names = ['Cat', 'Dog']
def plotData(generator, n_images):
    """
    Plots random data from dataset
    Args:
    generator: a generator instance
    n_images : number of images to plot
    """
    i = 1
    images, labels = next(generator)
    labels = labels.astype('int32')

    plt.figure(figsize=(14, 15))
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
        i += 1
        if i == n_images:
            break

    plt.show()

# plotData(train_generator, 10)
# plotData(validation_generator, 10)
# plotData(test_generator, 10)

#-----Model--------

inputs = tf.keras.layers.Input(shape=(150, 150, 3))
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(2,2)(x)

x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(2,2)(x)

x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(2,2)(x)

x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)


x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

if __name__ == "__main__":

    tf.debugging.set_log_device_placement(True)

    r = model.fit(
        train_generator,
        epochs=2,
        validation_data=validation_generator
    )

    if INCLUDE_TEST:
        model.evaluate(test_generator)

    model.save("cats_dogs_model.keras")

