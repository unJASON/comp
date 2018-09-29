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
import cv2
import numpy as np
import yaml


tagPath = 'Annotations/'
imgPath = 'BMPImages/'
imgObjMap = XmlProcess.extractAllXML(tagPath)
# print(imgObjMap)

model = Models.createYOLOModel()

# pre-operation of data
resImags, resAnswers = Models.mode(imgObjMap, imgPath)
# print(resAnswers.shape)
# 训练
# model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pre: y_pre}, metrics=['accuracy'])
def my_loss(y_true,y_pre):
    return y_pre
model.compile(optimizer=Adam(lr=1e-4), loss=my_loss, metrics=['accuracy'])
print("Training----")
for i in range(20):
    model.fit([resImags / 255., resAnswers], np.zeros(resAnswers.shape[0]), nb_epoch=5, batch_size=5)
    loss, accuracy = model.evaluate([resImags / 255., resAnswers], np.zeros(resAnswers.shape[0]))
    print('loss:',loss)
    if loss < 20:
        break
model.save_weights('model_25_weights.h5')
# print("Testing---------")
# testAnnoPath = 'testAnno/'
# testBmpPath = 'testBmp/'
# imgObjMapTest = XmlProcess.extractAllXML(testAnnoPath)
# testImages, testAnswers = Models.mode(imgObjMapTest, testBmpPath)
# # print(testImages.shape)
# test_Model = Model(inputs=model.input, outputs=model.get_layer('predict_layer').output)
# # plot_model(test_Model, to_file='model_test.png', show_shapes=True)
# data = test_Model.predict([testImages/255.0,testAnswers])
# print(Models.loss_v2(testAnswers, data))
# Models.showPredictExample(testImages,testAnswers,data)
# print("done")


########test#################
# test = cv2.cvtColor(np.asarray(Image.fromarray(np.uint8(resImags[0]))), cv2.COLOR_RGB2BGR)
# pc = resAnswers[0]
# for i in range(pc.shape[0]):
#     for j in range(pc.shape[1]):
#         if pc[i][j][0] == 1:
#             cv2.circle(test,(int(i*boxSize+pc[i][j][1]*boxSize),int(j*boxSize+pc[i][j][2]*boxSize)),radius=5,thickness=8,color=(255,255,255) )
#
# cv2.imshow('pic', test)
# cv2.waitKey(0)
########################test##################
