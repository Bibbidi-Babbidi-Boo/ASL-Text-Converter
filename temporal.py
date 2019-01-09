import numpy as np
import keras
from keras import backend as K
from keras.layers import GaussianNoise,GaussianDropout
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import save_img
import scipy.misc

path = '/home/rishab/IIT_M_Internship/ASL/asl_alphabet_train'
v_path = '/home/rishab/IIT_M_Internship/ASL/asl_alphabet_valid'
t_path = '/home/rishab/IIT_M_Internship/ASL/asl_alphabet_test'

batches = ImageDataGenerator().flow_from_directory(path, target_size=(224,224), classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','space','T','U','V','W','X','Y','Z'], shuffle=True, batch_size=32)
v_batches = ImageDataGenerator().flow_from_directory(v_path, target_size=(224,224), classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','space','T','U','V','W','X','Y','Z'], shuffle=True, batch_size=32)
t_batches = ImageDataGenerator().flow_from_directory(t_path, target_size=(224,224), classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','space','T','U','V','W','X','Y','Z'], shuffle=True, batch_size=120)
t_imgs,t_labels = next(t_batches)

img_input = Input(shape=(224,224,3))
#image = GaussianNoise(0.01)(img_input)
image = BatchNormalization()(img_input)
image = Conv2D(16, (1,1), strides=(2,2), activation='relu')(image)
image = Conv2D(16, (5,5), strides=(2,2), activation='relu')(image)
image = MaxPooling2D(pool_size=(3,3), strides=(2,2))(image)
image = Conv2D(24, (1,1), strides=(1,1), activation='relu')(image)
image = Conv2D(24, (3,3), strides=(1,1), activation='relu')(image)
image = MaxPooling2D(pool_size=(3,3), strides=(2,2))(image)
image = Conv2D(24, (3,3), strides=(2,2), activation='relu')(image)
image = BatchNormalization()(image)



tower_1a1 = Conv2D(8, (1,1), activation='relu', padding='same')(image)

tower_1a2 = Conv2D(12, (1,1), activation='relu', padding='same')(image)
tower_1a2 = Conv2D(16, (3,3), activation='relu', padding='same')(tower_1a2)

tower_1a3 = Conv2D(2, (1,1), activation='relu', padding='same')(image)
tower_1a3 = Conv2D(4, (5,1), activation='relu', padding='same')(tower_1a3)
tower_1a3 = Conv2D(4, (1,5), activation='relu', padding='same')(tower_1a3)

tower_1a4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(image)
tower_1a4 = Conv2D(4, (1,1), activation='relu', padding='same')(tower_1a4)

t_1 = keras.layers.concatenate([tower_1a1,tower_1a2,tower_1a3,tower_1a4], axis = 3)
t_1 = BatchNormalization()(t_1)
#t_1 = GaussianDropout(0.35)(t_1)

"""
tower_2a1 = Conv2D(12, (1,1), activation='relu', padding='same')(t_1)

tower_2a2 = Conv2D(8, (1,1), activation='relu', padding='same')(t_1)
tower_2a2 = Conv2D(12, (3,3), activation='relu', padding='same')(tower_2a2)

tower_2a3 = Conv2D(8, (1,1), activation='relu', padding='same')(t_1)
tower_2a3 = Conv2D(12, (3,3), activation='relu', padding='same')(tower_2a3)
tower_2a3 = Conv2D(12, (3,3), activation='relu', padding='same')(tower_2a3)

tower_2a4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(t_1)
tower_2a4 = Conv2D(12, (1,1), activation='relu', padding='same')(tower_2a4)

t_2 = keras.layers.concatenate([tower_2a1,tower_2a2,tower_2a3,tower_2a4], axis = 3)
t_2 = BatchNormalization()(t_2)
#t_2 = GaussianDropout(0.35)(t_2)


tower_3a1 = Conv2D(24, (1,1), activation='relu', padding='same')(t_2)
tower_3a1 = Conv2D(24, (3,3), activation='relu')(tower_3a1)

tower_3a2 = Conv2D(32, (1,1), activation='relu', padding='same')(t_2)
tower_3a2 = Conv2D(32, (7,1), activation='relu', padding='same')(tower_3a2)
tower_3a2 = Conv2D(40, (1,7), activation='relu', padding='same')(tower_3a2)
tower_3a2 = Conv2D(40, (3,3), activation='relu')(tower_3a2)

t_3 = keras.layers.concatenate([tower_3a1,tower_3a2], axis = 3)
t_3 = BatchNormalization()(t_3)
#t_3 = GaussianDropout(0.35)(t_3)



tower_4a1 = Conv2D(16, (1,1), activation='relu', padding='same')(t_3)

tower_4a2 = Conv2D(16, (1,1), activation='relu', padding='same')(t_3)
tower_4a2 = Conv2D(32, (3,3), activation='relu', padding='same')(tower_4a2)#20

tower_4a3 = Conv2D(3, (1,1), activation='relu', padding='same')(t_3)
tower_4a3 = Conv2D(8, (5,1), activation='relu', padding='same')(tower_4a3)
tower_4a3 = Conv2D(8, (1,5), activation='relu', padding='same')(tower_4a3)

tower_4a4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(t_3)
tower_4a4 = Conv2D(8, (1,1), activation='relu', padding='same')(tower_4a4)

t_4 = keras.layers.concatenate([tower_4a1,tower_4a2,tower_4a3,tower_4a4], axis = 3)
t_4 = BatchNormalization()(t_4)
#t_4 = GaussianDropout(0.35)(t_4)


tower_5a1 = Conv2D(48, (3,3), activation='relu', strides=(2,2))(t_4)

tower_5a2 = Conv2D(24, (1,1), activation='relu', padding='same')(t_4)
tower_5a2 = Conv2D(28, (3,3), activation='relu', padding='same')(tower_5a2)
tower_5a2 = Conv2D(32, (3,3), activation='relu' , strides=(2,2))(tower_5a2)

tower_5a3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(t_4)

t_5 = keras.layers.concatenate([tower_5a1,tower_5a2,tower_5a3], axis = 3)
t_5 = BatchNormalization()(t_5)
#t_5 = GaussianDropout(0.35)(t_5)


tower_6a = Conv2D(32, (1,1), activation='relu', padding='same')(t_5)

tower_6b = Conv2D(46, (1,1), activation='relu', padding='same')(t_5)
tower_6b = Conv2D(56, (1,3), activation='relu', padding='same')(tower_6b)
tower_6b = Conv2D(64, (3,1), activation='relu', padding='same')(tower_6b)
tower_6b1 = Conv2D(32, (1,3), activation='relu', padding='same')(tower_6b)
tower_6b2 = Conv2D(32, (3,1), activation='relu', padding='same')(tower_6b)

tower_6c = Conv2D(48, (1,1), activation='relu', padding='same')(t_5)
tower_6c1 = Conv2D(32, (1,3), activation='relu', padding='same')(tower_6c)
tower_6c2 = Conv2D(32, (3,1), activation='relu', padding='same')(tower_6c)

tower_6d = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(t_5)
tower_6d = Conv2D(32, (1,1), activation='relu', padding='same')(tower_6d)

t_6 = keras.layers.concatenate([tower_6a,tower_6b1,tower_6b2,tower_6c1,tower_6c2,tower_6d], axis = 3)
t_6 = BatchNormalization()(t_6)
#t_6 = GaussianDropout(0.35)(t_6)

tower_7a1 = Conv2D(32, (1,1), activation='relu', padding='same')(t_6)
tower_7a1 = Conv2D(32, (1,7), activation='relu', padding='same')(tower_7a1)
tower_7a1 = Conv2D(40, (7,1), activation='relu', padding='same')(tower_7a1)
tower_7a1 = Conv2D(40, (3,3), activation='relu' , strides=(2,2))(tower_7a1)

tower_7a2 = Conv2D(24, (1,1), activation='relu', padding='same')(t_6)
tower_7a2 = Conv2D(24, (3,3), activation='relu' , strides=(2,2))(tower_7a2)

tower_7a3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(t_6)

t_7 = keras.layers.concatenate([tower_7a1,tower_7a2,tower_7a3], axis = 3)
t_7 = BatchNormalization()(t_7)
"""

out = GlobalAveragePooling2D(data_format='channels_last')(t_1)
out = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.35))(out)
out = GaussianDropout(0.65)(out, training=True)
out = Dense(27, activation='softmax',kernel_regularizer=regularizers.l2(0.35))(out)

"""
out = Flatten()(t_7)
out = GaussianDropout(0.35)(out)
out = Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.032))(out)
out = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.032))(out)
out = Dense(27, activation='softmax',kernel_regularizer=regularizers.l2(0.032))(out)
"""
model = Model(img_input, out)
print(model.summary())

model.compile(Adam(lr=0.002,decay=2.5e-8), loss='categorical_crossentropy', metrics=['accuracy'])

filepath="/home/rishab/one_incept.hdf5"
#model.load_weights('/home/rishab/temporal.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [EarlyStopping(monitor='val_acc', patience=4, verbose=1),ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

history = model.fit_generator(batches, steps_per_epoch=2483, validation_data=v_batches, validation_steps=719, epochs=5, verbose=1, callbacks=callbacks_list)

y_pred = model.predict(t_imgs,steps=1)
score = model.evaluate(t_imgs, t_labels, verbose=1)
print(score)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy one inception')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss one inception')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.show()

K.clear_session()
