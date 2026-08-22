from main import validation_generator, class_names, r

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = tf.keras.models.load_model("cats_dogs_model.keras")

def plot_prediction(generator, n_images):
    """
    Test the model on random predictions
    Args:
    generator: a generator instance
    n_images : number of images to plot

    """
    i = 1
    images, labels = next(generator)
    preds = model.predict(images)
    predictions = np.argmax(preds, axis=1)
    labels = labels.astype('int32')
    plt.figure(figsize=(14, 15))
    for i, (image, label) in enumerate(zip(images, labels)):
        if i == n_images:
            break

        plt.subplot(5, 4, i+1)
        plt.imshow(image)
        if predictions[i] == label:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='g') 
            plt.axis('off')
        else:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='r') 
            plt.axis('off')
       

    plt.show()

plot_prediction(validation_generator, 20)
# plot_prediction(train_generator, 21)

results = pd.DataFrame(r.history)
results.tail()