import keras
import XmlProcess
import Models
import PIL.Image as Image
# import matplotlib.pyplot as pyplot
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers import Input, Lambda
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf


def my_loss(y_true, y_pre):
    return y_pre


tagPath = 'Annotations/'
imgPath = 'BMPImages/'
imgObjMap = XmlProcess.extractAllXML(tagPath)
resImags, resAnswers = Models.mode(imgObjMap, imgPath)
model = Models.createYOLOModel()
model.load_weights('model_25_weights.h5', by_name=True)

model.compile(optimizer=Adam(lr=1e-4), loss=my_loss, metrics=['accuracy'])
print("Training----")
for i in range(10):
    model.fit([resImags / 255, resAnswers], np.zeros(resAnswers.shape[0]), nb_epoch=5, batch_size=5)
    loss, accuracy = model.evaluate([resImags / 255, resAnswers], np.zeros(resAnswers.shape[0]))
    print('loss:',loss)
    if loss < 15:
        break

# model.save('model_25.h5')
model.save_weights('model_25_weights.h5')
