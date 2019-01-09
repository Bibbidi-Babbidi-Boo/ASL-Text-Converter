import numpy as np
import keras
from keras import backend as K
from keras.layers import GaussianNoise,GaussianDropout
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import SeparableConv2D, MaxPooling2D, Activation, Concatenate, AveragePooling2D, Dropout, GlobalAveragePooling2D
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
import time


img_input = Input(shape=(224,224,3))
image = GaussianNoise(0.01)(img_input)
image = BatchNormalization()(img_input)
image = SeparableConv2D(16, (1,1), strides=(2,2), activation='relu')(image)
image = SeparableConv2D(16, (5,5), strides=(2,2), activation='relu')(image)
image = MaxPooling2D(pool_size=(3,3), strides=(2,2))(image)
image = SeparableConv2D(32, (3,3), strides=(2,2), activation='relu')(image)
image = BatchNormalization()(image)


tower_1a1 = SeparableConv2D(12, (1,1), activation='relu', padding='same')(image)

tower_1a2 = SeparableConv2D(8, (1,1), activation='relu', padding='same')(image)
tower_1a2 = SeparableConv2D(12, (3,3), activation='relu', padding='same')(tower_1a2)

tower_1a3 = SeparableConv2D(8, (1,1), activation='relu', padding='same')(image)
tower_1a3 = SeparableConv2D(12, (3,3), activation='relu', padding='same')(tower_1a3)
tower_1a3 = SeparableConv2D(12, (3,3), activation='relu', padding='same')(tower_1a3)

tower_1a4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(image)
tower_1a4 = SeparableConv2D(12, (1,1), activation='relu', padding='same')(tower_1a4)

t_1 = keras.layers.concatenate([tower_1a1,tower_1a2,tower_1a3,tower_1a4], axis = 3)
t_1 = BatchNormalization()(t_1)
#t_1 = GaussianDropout(0.2)(t_1)


tower_2a1 = SeparableConv2D(24, (1,1), activation='relu', padding='same')(t_1)
tower_2a1 = SeparableConv2D(24, (3,3), activation='relu')(tower_2a1)

tower_2a2 = SeparableConv2D(32, (1,1), activation='relu', padding='same')(t_1)
tower_2a2 = SeparableConv2D(32, (7,1), activation='relu', padding='same')(tower_2a2)
tower_2a2 = SeparableConv2D(40, (1,7), activation='relu', padding='same')(tower_2a2)
tower_2a2 = SeparableConv2D(40, (3,3), activation='relu')(tower_2a2)

t_2 = keras.layers.concatenate([tower_2a1,tower_2a2], axis = 3)
t_2 = BatchNormalization()(t_2)
#t_2 = GaussianDropout(0.2)(t_2)

tower_3a1 = SeparableConv2D(48, (3,3), activation='relu', strides=(2,2))(t_2)

tower_3a2 = SeparableConv2D(24, (1,1), activation='relu', padding='same')(t_2)
tower_3a2 = SeparableConv2D(28, (3,3), activation='relu', padding='same')(tower_3a2)
tower_3a2 = SeparableConv2D(32, (3,3), activation='relu' , strides=(2,2))(tower_3a2)

tower_3a3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(t_2)

t_3 = keras.layers.concatenate([tower_3a1,tower_3a2,tower_3a3], axis = 3)
t_3 = BatchNormalization()(t_3)
#t_3 = GaussianDropout(0.2)(t_3)


tower_4a = SeparableConv2D(32, (1,1), activation='relu', padding='same')(t_3)

tower_4b = SeparableConv2D(46, (1,1), activation='relu', padding='same')(t_3)
tower_4b = SeparableConv2D(56, (1,3), activation='relu', padding='same')(tower_4b)
tower_4b = SeparableConv2D(64, (3,1), activation='relu', padding='same')(tower_4b)
tower_4b1 = SeparableConv2D(32, (1,3), activation='relu', padding='same')(tower_4b)
tower_4b2 = SeparableConv2D(32, (3,1), activation='relu', padding='same')(tower_4b)

tower_4c = SeparableConv2D(48, (1,1), activation='relu', padding='same')(t_3)
tower_4c1 = SeparableConv2D(32, (1,3), activation='relu', padding='same')(tower_4c)
tower_4c2 = SeparableConv2D(32, (3,1), activation='relu', padding='same')(tower_4c)

tower_4d = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(t_3)
tower_4d = SeparableConv2D(32, (1,1), activation='relu', padding='same')(tower_4d)

t_4 = keras.layers.concatenate([tower_4a,tower_4b1,tower_4b2,tower_4c1,tower_4c2,tower_4d], axis = 3)
t_4 = BatchNormalization()(t_4)
#t_4 = GaussianDropout(0.2)(t_4)

tower_5a1 = SeparableConv2D(32, (1,1), activation='relu', padding='same')(t_4)
tower_5a1 = SeparableConv2D(32, (1,7), activation='relu', padding='same')(tower_5a1)
tower_5a1 = SeparableConv2D(40, (7,1), activation='relu', padding='same')(tower_5a1)
tower_5a1 = SeparableConv2D(40, (3,3), activation='relu' , strides=(2,2))(tower_5a1)

tower_5a2 = SeparableConv2D(24, (1,1), activation='relu', padding='same')(t_4)
tower_5a2 = SeparableConv2D(24, (3,3), activation='relu' , strides=(2,2))(tower_5a2)

tower_5a3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(t_4)

t_5 = keras.layers.concatenate([tower_5a1,tower_5a2,tower_5a3], axis = 3)
t_5 = BatchNormalization()(t_5)
#t_5 = GaussianDropout(0.2)(t_5)


out = GlobalAveragePooling2D(data_format='channels_last')(t_5)
out = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.215))(out)
out = GaussianDropout(0.75)(out)
out = Dense(27, activation='softmax',kernel_regularizer=regularizers.l2(0.215))(out)


model = Model(img_input, out)
print(model.summary())

model.compile(Adam(lr=0.002,decay=2.5e-8), loss='categorical_crossentropy', metrics=['accuracy'])

filepath="/home/rishab/temporal.hdf5"
model.load_weights('/home/rishab/temporal.hdf5')

haar_face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
if haar_face_cascade.empty():
    print("EMPTY")

def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

def FindDistance(A,B):
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))


capa = cv2.VideoCapture(0)

p=0
lower_skin = np.ndarray(shape=(1,3))
upper_skin = np.ndarray(shape=(1,3))
px = np.ndarray(shape=(4,3))
pixel_min = np.ndarray(shape=(21,3))
pixel_max = np.ndarray(shape=(21,3))

times=0

print("Ready")

k=0
start_time = time.time()
while(True):
    #print(int(time.time() - start_time))
    #if times == 0 or int((time.time() - start_time))%3 == 0:
    #    ret33,f = capa.read()
    #    print("inside nowq")
    #times+=1
    ret2, frame2 = capa.read()
    ret, frame = capa.read()
    b = frame.shape[1]
    frame[0:160, 0:225] = [0,0,0]
    frame[0:160, b-225:b] = [0,0,0]
    frame = cv2.flip(frame,1)
    frame2 = cv2.flip(frame2,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(frame,(11,11),0)
    blur = cv2.medianBlur(blur,5)
    ret,thresh_frame1 = cv2.threshold(gray,205,255,cv2.THRESH_BINARY)
    cv2.imshow('thresh_frame1', thresh_frame1)
    thresh_frame1_inv = cv2.bitwise_not(thresh_frame1)
    #cv2.imshow('thresh_frame_inv', thresh_frame1_inv)
    ret,thresh_frame2 = cv2.threshold(gray,220,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh_frame2', thresh_frame2)
    th = thresh_frame1 - thresh_frame2
    #cv2.imshow('th', th)
    #thresh_frame = cv2.bitwise_and(thresh_frame1, thresh_frame2)
    #thresh_frame = cv2.dilate(thresh_frame,(3,3),iterations = 1)
    #thresh_frame = cv2.erode(thresh_frame,(3,3),iterations = 1)
    #cv2.imshow('thresh_frame', thresh_frame)
    th = cv2.bitwise_not(th)
    cv2.imshow('th', th)

    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = frame[y+2:y+h-2, x+2:x+w-2]
        hsv_frame = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)
        w_f = hsv_frame.shape[0]
        h_f = hsv_frame.shape[1]
        cv2.circle(roi_color, ((w/2).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        #cv2.circle(roi_color, ((w/4).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/5).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/4).astype(int),(h/3).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/2).astype(int),(h/8).astype(int)), 3, (0,255,0), -1)
        if p <= 20:
            px[0] = hsv_frame[int(w/2), int(h/2)]
            px[1] = hsv_frame[int(w/5), int(h/2)]
            px[2] = hsv_frame[int(w/4), int(h/3)]
            px[3] = hsv_frame[int(w/2), int(h/8)]
            pixel_max[p] = np.amax(px, axis=0)
            pixel_min[p] = np.amin(px, axis=0)
            print("MAX",pixel_max[p])
            print("MIN",pixel_min[p])
        if p == 21:
            lower_skin = np.mean(pixel_min, axis=0)
            lower_skin[1] = lower_skin[1]-4
            lower_skin[2] = lower_skin[2]-4
            print(lower_skin)
            upper_skin = np.amax(pixel_max, axis=0)
            print(upper_skin)

        #print(p)
        if p > 21:

            #print("Mask ready")
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            cv2.imshow('premask', mask)
            #mask = mask-f
            #mask = mask - thresh_frame1

            #mask = cv2.bitwise_and(mask, thresh_frame2)
            #mask = cv2.medianBlur(mask,3)

            #mask = cv2.medianBlur(mask,25)              ``

            mask = cv2.erode(mask,(5,5),iterations = 8)
            mask = cv2.dilate(mask,(5,5),iterations = 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.dilate(mask, kernel)


            #size = mask.shape[1]
            mask[y-15:y+h+150, x-5:x+w+10] = 0
            #mask[0:10, 0:10] = 0
            cv2.imshow('mask', mask)

            mask2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                j=0
                max = 0
                pos = 0
                for j in range(len(contours)):
                    #print("j", j)
                    cnt1 = contours[j]
                    area = cv2.contourArea(cnt1)
                    #print(area)
                    if area > max:
                        max = area
                        pos = j

                cnt = contours[pos]
                perimeter = cv2.arcLength(cnt,True)
                #print("PERI",perimeter)

                cv2.drawContours(frame, cnt, -1, (0,0,255), 3)
                #cv2.imshow('mask2', mask2)
                hull = cv2.convexHull(cnt)
                hull2 = cv2.convexHull(cnt,returnPoints = False)
                defects = cv2.convexityDefects(cnt,hull2)
                FarDefect = []
                if defects is None:
                    break

                else:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        FarDefect.append(far)
                        cv2.line(frame,start,end,[0,255,0],1)
                        cv2.circle(frame,far,7,[100,255,255],1)
                    moments = cv2.moments(cnt)
                    if moments['m00']!=0:
                        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
                    centerMass=(cx,cy)
                    cv2.circle(frame,centerMass,7,[100,0,255],2)
                    final = cv2.drawContours(frame, hull, -1, (255,0,0), 3)
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w+20,y+h+20),(0,255,255),2)
                    if perimeter >= 500:
                        frame_roi = frame2[y-30:y+h, x:x+w]
                        resize = cv2.resize(frame_roi,(224,224), interpolation = cv2.INTER_CUBIC)

                        #mask11 = np.zeros(resize.shape[:2],np.uint8)
                        #bgdModel = np.zeros((1,65),np.float64)
                        #fgdModel = np.zeros((1,65),np.float64)
                        #rect = (50,50,450,290)
                        #cv2.grabCut(resize,mask11,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                        #mask21 = np.where((mask11==2)|(mask11==0),0,1).astype('uint8')
                        #resize = resize*mask21[:,:,np.newaxis]
                        cv2.imshow('frame_roi', resize)
                        if cv2.waitKey(1) & 0xFF == ord('b'):
                            break
                        resize = np.expand_dims(resize,axis=0)

                        #resize = prepare_img(resize)

                        y_pred = model.predict(resize,steps=1)
                        print(y_pred)
                        #start_time = time.time()
                        #while time.time() - start_time < 5:
                        #    continuevalid
                        #print(np.sum(y_pred))
                        h=0
                        max=0
                        for h in range (27):
                            if y_pred[0][h] > max:

                                max = y_pred[0][h]
                                pos = h
                                print(h)

                        labs_pred = pos+1
                        print("Predicted",labs_pred)
                        print("  ")

    cv2.imshow('frame', frame)


    p+=1
    #print(p)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capa.release()
cv2.destroyAllWindows()
