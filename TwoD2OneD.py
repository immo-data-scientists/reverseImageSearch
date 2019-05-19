import tensorflow as tf
from keras.preprocessing import image
import numpy as np

def normaliseImage(image):
    # image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image = tf.image.rgb_to_grayscale(image) # image /= 255.0
    return image 

def twoD2oneD(imgs):
    trainingData = []
    imageCount = len(imgs)
    for iImg in range(imageCount):
        img = image.load_img(imgs[:iImg+1][0], target_size=(224, 224))
        # image = tf.image.rgb_to_grayscale(image)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # img = tf.reshape(img, [-1])
        # print(img)
        trainingData.append(img)
    return trainingData




