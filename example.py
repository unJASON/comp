import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.datasets import mnist
from keras.utils import plot_model
from PIL import Image

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,1,28,28)/255.
X_test = X_test.reshape(-1,1,28,28)/255.
y_train = np_utils.to_categorical(y_train,num_classes= 10)
y_test = np_utils.to_categorical(y_test,num_classes= 10)

print(X_test.shape)
print(y_train.shape)
model = Sequential()


model.add(Convolution2D(
    filters = 32,
    kernel_size = (5,5),
    padding = 'same',
    data_format= 'channels_first',
    input_shape = (1,   #channel
                   28,  28), #height& width
    activation='relu'
))
# model.add(Activation('relu'))
model.add(MaxPooling2D(
    strides = (2,2),
    padding = 'same',
))

model.add(Convolution2D(64,kernel_size = (5,5),padding = 'same',activation='relu'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2),padding = 'same'))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
#model.add(Activation('relu'))
model.add(Dense(10,activation='softmax'))
#model.add(Activation('softmax'))
adam = Adam(lr = 1e-4)

#showImg
#plot_model(model, to_file='model.png')


model.compile(optimizer= adam, loss='categorical_crossentropy',metrics=['accuracy'])
print("Training----")
model.fit(X_train,y_train,nb_epoch=50,batch_size= 32)
print("Testing---------")
loss,accuracy = model.evaluate(X_test,y_test)
print('\ntest loss: ',loss)
print('\n test accuracy: ',accuracy)

