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
# from keras.utils import plot_model
from keras.models import load_model



def my_loss(y_true, y_pre):
    return y_pre


model = Models.createYOLOModel()
model.load_weights('model_25_weights.h5',by_name=True)
# model = load_model('model_25.h5', custom_objects={'my_loss': my_loss})
print("Testing---------")

testAnnoPath = 'testAnno/'
testBmpPath = 'testBmp/'
imgObjMapTest = XmlProcess.extractAllXML(testAnnoPath)
testImages, testAnswers = Models.mode(imgObjMapTest, testBmpPath)
test_Model = Model(inputs=model.input, outputs=model.get_layer('predict_layer').output)
# plot_model(test_Model, to_file='model_test.png', show_shapes=True)


data = test_Model.predict([testImages/255.0, testAnswers])
for l in range(testAnswers.shape[0]):
    print(Models.loss_v2(testAnswers[l:l+1],data[l:l+1]))
Models.showPredictExample(testImages,testAnswers, data)

print("done")
