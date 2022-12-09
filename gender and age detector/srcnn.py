import os
import sys
import keras
import cv2
import numpy as np
import skimage
import streamlit as st
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

#from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim


#from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
#from keras.applications import ImageDataGenerator, array_to_img, img_to_array


# define the SRCNN model
@st.cache(suppress_st_warning=True)
def model():
    
    # define model type
    SRCNN = Sequential()
    
    #add model layers;filters =no. of nodes in the layer
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))#only if in keras.json image_data_format is channels_last; else if channels_first then 1,None,None
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    #input_shape takes image of any height and width as long it is one channel
    #that is how the SRCNN handles input,it handles image slice inputs, it doesn't work at all 3 channels at once
    #SRCNN was trained on the luminescence channel in the YCrCb color space 
    
    # define optimizer
    adam = Adam(lr=0.0003)
    
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN

SRCNN = model()


load_weights = 'model_5545_993_20_10.h5'
#st.write(load_weights)
SRCNN.load_weights(load_weights)
print("weights loaded")

def predictCNN(input_img):
    scale = 2

    img = np.array(input_img)

    #st.write(type(img))
    
    shape= img.shape
    #st.write(shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_img = cv2.resize(img[:, :, 0], (int (shape[1] * scale), int (shape[0] * scale)), cv2.INTER_CUBIC)
    img = cv2.resize(img, (int (shape[1] * scale), int (shape[0] * scale)), cv2.INTER_CUBIC)

    print(img.shape)
    print(Y_img.shape)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    #st.write("INPUT")
    #st.image(img)

    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.

    pre = SRCNN.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]

    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    #st.write("OUTPUT")
    #st.image(img)
    cv2.imwrite("results/restored_imgs/crop_img_0_cnn.png",img)
    return img
