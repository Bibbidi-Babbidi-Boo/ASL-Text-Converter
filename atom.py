import numpy as np
import keras
from keras import backend as K
from keras.layers import GaussianNoise,GaussianDropout,Dropout,SeparableConv2D, MaxPooling2D, Activation, Concatenate, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import Input
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import save_img
import scipy.misc
import time
from keras.constraints import max_norm
from timeit import default_timer as timer


global flag, now, now_p, count_pred, count, change, background


background = cv2.imread('background.png')
#cv2.imshow("background", background)
background = cv2.cvtColor(background,cv2.COLOR_BGR2HSV)
flag = 1
change = 0
now = timer()
now_p = timer()
count = 0
count_pred = []
word = ""

def most_frequent(List): 
    return max(set(List), key = List.count) 

img_input = Input(shape=(244,244,3))
image = BatchNormalization()(img_input)
image = Conv2D(16, (3,3), strides=(2,2), activation='relu')(image)
image_1 = Conv2D(32, (3,3), strides=(2,2), activation='relu')(image)

image_2 = Conv2D(16, (1,1), activation='relu', padding='same')(image_1)
image_3 = Conv2D(32, (3,3), activation='relu', padding='same')(image_2)
image = keras.layers.add([image_1,image_3])

image_4 = Conv2D(64, (3,3), activation='relu', padding='same')(image)

image_5 = Conv2D(32, (1,1), activation='relu', padding='same')(image_4)
image_6 = Conv2D(64, (3,3), activation='relu', padding='same')(image_5)
image = keras.layers.add([image_4,image_6])
image = BatchNormalization()(image)



tower_a1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(image)
tower_a1 = Conv2D(32, (3,3), activation='relu')(tower_a1)

tower_a2 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_a2 = Conv2D(32, (3,3), activation='relu', padding='same')(tower_a2)
tower_a2 = Conv2D(32, (3,3), activation='relu')(tower_a2)

tower_a3 = Conv2D(40, (1,1), activation='relu', padding='same')(image)
tower_a3 = Conv2D(40, (3,3), activation='relu')(tower_a3)

t_a = keras.layers.concatenate([tower_a1,tower_a2,tower_a3], axis = 3)
t_a = BatchNormalization()(t_a)




tower_11 = Conv2D(128, (1,1), activation='relu', padding='same')(t_a)

tower_12 = Conv2D(64, (1,1), activation='relu', padding='same')(t_a)
tower_12 = Conv2D(128, (3,3), activation='relu', padding='same')(tower_12)

tower_13 = Conv2D(32, (1,1), activation='relu', padding='same')(t_a)
tower_13 = Conv2D(64, (3,3), activation='relu', padding='same')(tower_13)
tower_13 = Conv2D(128, (3,3), activation='relu', padding='same')(tower_13)

t_1 = keras.layers.add([tower_11, tower_12, tower_13])
t_1 = BatchNormalization()(t_1)



image_10 = Conv2D(128, (3,3), strides=(2,2), activation='relu')(t_1)

image_11 = Conv2D(64, (1,1), activation='relu', padding='same')(image_10)
image_12 = Conv2D(128, (3,3), activation='relu', padding='same')(image_11)
image_13 = keras.layers.add([image_10,image_12])
image_14 = Conv2D(64, (1,1), activation='relu', padding='same')(image_13)
image_15 = Conv2D(128, (3,3), activation='relu', padding='same')(image_14)
image = keras.layers.add([image_13,image_15])        
image = BatchNormalization()(image)



tower_b1 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_b1 = Conv2D(64, (3,3), activation='relu')(tower_b1)

tower_b2 = Conv2D(16, (1,1), activation='relu', padding='same')(image)
tower_b2 = Conv2D(32, (5,5), activation='relu', padding='same')(tower_b2)
tower_b2 = Conv2D(64, (3,3), activation='relu')(tower_b2)

tower_b3 = Conv2D(16, (1,1), activation='relu', padding='same')(image)
tower_b3 = Conv2D(16, (1,7), activation='relu', padding='same')(tower_b3)
tower_b3 = Conv2D(16, (7,1), activation='relu', padding='same')(tower_b3)
tower_b3 = Conv2D(64, (3,3), activation='relu')(tower_b3)

tower_b4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(image)
tower_b4 = Conv2D(64, (3,3), activation='relu')(tower_b4)

t_b = keras.layers.concatenate([tower_b1,tower_b2,tower_b3,tower_b4], axis = 3)
t_b = BatchNormalization()(t_b)




tower_21 = Conv2D(128, (1,1), activation='relu', padding='same')(t_b)
tower_21 = Conv2D(256, (3,3), activation='relu')(tower_21)            ####same padding
  
tower_22 = Conv2D(32, (1,1), activation='relu', padding='same')(t_b)
tower_22 = Conv2D(128, (1,7), activation='relu', padding='same')(tower_22)
tower_22 = Conv2D(128, (7,1), activation='relu', padding='same')(tower_22)
tower_22 = Conv2D(256, (3,3), activation='relu')(tower_22)

t_2 = keras.layers.add([tower_21, tower_22])
t_2 = BatchNormalization()(t_2)



image_35 = Conv2D(256, (3,3), strides=(2,2), activation='relu')(t_2)

image_36 = Conv2D(128, (1,1), activation='relu', padding='same')(image_35)
image_37 = Conv2D(256, (3,3), activation='relu', padding='same')(image_36)
image_38 = keras.layers.add([image_35,image_37])
image_39 = Conv2D(128, (1,1), activation='relu', padding='same')(image_38)
image_40 = Conv2D(256, (3,3), activation='relu', padding='same')(image_39)
image = keras.layers.add([image_38,image_40])
image = BatchNormalization()(image)



tower_c1 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_c1 = Conv2D(64, (3,3), activation='relu')(tower_c1)

tower_c21 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_c21 = Conv2D(32, (1,3), activation='relu', padding='same')(tower_c21)
tower_c21 = Conv2D(64, (3,3), activation='relu')(tower_c21)

tower_c22 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_c22 = Conv2D(32, (3,1), activation='relu', padding='same')(tower_c22)
tower_c22 = Conv2D(64, (3,3), activation='relu')(tower_c22)

tower_c31 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_c31 = Conv2D(32, (3,3), activation='relu', padding='same')(tower_c31)
tower_c31 = Conv2D(32, (3,1), activation='relu', padding='same')(tower_c31)
tower_c31 = Conv2D(64, (3,3), activation='relu')(tower_c31)

tower_c32 = Conv2D(32, (1,1), activation='relu', padding='same')(image)
tower_c32 = Conv2D(32, (3,3), activation='relu', padding='same')(tower_c32)
tower_c32 = Conv2D(32, (1,3), activation='relu', padding='same')(tower_c32)
tower_c32 = Conv2D(64, (3,3), activation='relu')(tower_c32)

tower_c4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(image)
tower_c4 = Conv2D(64, (3,3), activation='relu')(tower_c4)

t_c = keras.layers.concatenate([tower_c1,tower_c21,tower_c22,tower_c31,tower_c32,tower_c4], axis = 3)
t_c = BatchNormalization()(t_c)



tower_31 = Conv2D(256, (1,1), activation='relu', padding='same')(t_c)
tower_31 = Conv2D(400, (3,3), activation='relu')(tower_31)                ###Same padding

tower_32 = Conv2D(128, (1,1), activation='relu', padding='same')(t_c)
tower_32 = Conv2D(256, (1,7), activation='relu', padding='same')(tower_32)
tower_32 = Conv2D(256, (7,1), activation='relu', padding='same')(tower_32)
tower_32 = Conv2D(400, (3,3), activation='relu')(tower_32)

t_3 = keras.layers.add([tower_31, tower_32])
t_3 = BatchNormalization()(t_3)



out = GlobalAveragePooling2D(data_format='channels_last')(t_3)
out = Dense(100, activation='relu',kernel_constraint=max_norm(3),kernel_regularizer=regularizers.l2(0.65))(out)
out = Dense(27, activation='softmax',kernel_constraint=max_norm(3),kernel_regularizer=regularizers.l2(0.65))(out)


model = Model(img_input, out)
print(model.summary())

model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('/home/rishab/Documents/IIT_M/yolo_resnet4.hdf5')

haar_face_cascade = cv2.CascadeClassifier('/home/rishab/opencv-3.4.5/data/haarcascades/haarcascade_frontalface_default.xml')
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
    global flag, now, now_p, count, count_pred, change, background, word
    font = cv2.FONT_HERSHEY_SIMPLEX
    now = timer()

    ret2, frame2 = capa.read()
    ret, frame = capa.read()
    frame = cv2.flip(frame,1)
    frame2 = cv2.flip(frame2,1)

    cv2.imshow("Frame", frame2)
    b = frame.shape[1]
    frame[0:100, 0:b] = [0,0,0]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    faces = haar_face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=25)
    ret_t1,thresh_frame1 = cv2.threshold(gray2,180,255,cv2.THRESH_BINARY_INV) ##gap around 40
    ret_t2,thresh_frame2 = cv2.threshold(gray2,120,255,cv2.THRESH_BINARY)
    #ret_mask1, mask_back1 = cv2.threshold(background,140,255,cv2.THRESH_BINARY_INV)
    #ret_mask2, mask_back2 = cv2.threshold(background,190,255,cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh_frame1, thresh_frame2)
    cv2.imshow("threshold", thresh)

    for (x,y,w,h) in faces:
        w = w-5
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = frame[y+2:y+h-2, x+2:x+w-2]
        hsv_frame = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)
        w_f = hsv_frame.shape[0]
        h_f = hsv_frame.shape[1]
        cv2.circle(roi_color, ((w/2).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/4).astype(int),(1.6*h/3).astype(int)), 3, (0,255,0), -1)
        #cv2.circle(roi_color, ((w/6).astype(int),(1.25*h/3).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/2).astype(int),(h/8).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w-w/4).astype(int),(1.6*h/3).astype(int)), 3, (0,255,0), -1)
        #cv2.circle(roi_color, ((w-w/6).astype(int),(1.25*h/3).astype(int)), 3, (0,255,0), -1)
        if p <= 20:
            px[0] = hsv_frame[int(w/2), int(h/2)]
            px[1] = hsv_frame[int(w/4), int(1.6*h/3)]
            #px[2] = hsv_frame[int(w/6), int(1.25*h/3)]
            px[2] = hsv_frame[int(w/2), int(h/8)]
            px[3] = hsv_frame[int(w-w/4), int(1.6*h/3)]
            #px[5] = hsv_frame[int(w-w/6), int(1.25*h/3)]
            pixel_max[p] = np.amax(px, axis=0)
            pixel_min[p] = np.amin(px, axis=0)
            #print("MAX",pixel_max[p])
            #print("MIN",pixel_min[p])
        if p == 21:
            upper_skin = np.amax(pixel_max, axis=0)
            lower_skin = np.amin(pixel_min, axis=0)
            #lower_skin[0] = lower_skin[0]-2
            #lower_skin[1] = lower_skin[1]-2
            #lower_skin[2] = lower_skin[2]-2
            #upper_skin[0] = upper_skin[0]+2
            #upper_skin[1] = upper_skin[1]+2
            #upper_skin[2] = upper_skin[2]+2
            #print(lower_skin)
            #print(upper_skin)

        if p > 21:
            mask_back1 = cv2.inRange(background, lower_skin, upper_skin)
            mask_back1 = cv2.bitwise_not(mask_back1)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            cv2.imshow('mask1', mask)
            #cv2.imshow('mask2', mask_back1)
            mask = cv2.bitwise_and(mask, mask_back1)
            mask = cv2.bitwise_and(mask, thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.dilate(mask,kernel,iterations = 7)
            mask = cv2.erode(mask,kernel,iterations = 5)
            closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5)
            


            #mask = cv2.dilate(mask,kernel,iterations = 5)
            #mask = cv2.erode(mask,kernel,iterations = 3)
            #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 20)
            #cv2.imshow('premask', mask)
            #mask = cv2.erode(mask,(5,5),iterations = 8)
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            #   mask = cv2.dilate(mask,(5,5),iterations = 5)
            #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            #mask = cv2.dilate(mask, kernel)
            mask[y-50:y+h+150, x-2:x+w+20] = 0
            cv2.imshow('mask', mask)
            mask2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                j=0
                maximum = 0
                pos = 0
                for j in range(len(contours)):
                    cnt1 = contours[j]
                    area = cv2.contourArea(cnt1)
                    if area > maximum:
                        maximum = area
                        pos = j

                cnt = contours[pos]
                perimeter = cv2.arcLength(cnt,True)
                area = cv2.contourArea(cnt)
                cv2.drawContours(frame, cnt, -1, (0,0,255), 3)
                hull = cv2.convexHull(cnt)
                hull2 = cv2.convexHull(cnt,returnPoints = False)
                defects = cv2.convexityDefects(cnt,hull2)
                FarDefect = []
                if defects is None:
                    break
                else:

                    #final = cv2.drawContours(frame, hull, -1, (255,0,0), 3)
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w+30,y+h+30),(0,255,255),2)
                    
                    if perimeter >= 500 and area>3200:
                        print("AREA", area)
                        #print("YES")
                        #print(now)
                        if x<30:
                            if y<30:
                                frame_roi = frame2[0:y+h+30, 0:x+w+30]
                            elif y+h>450:
                                frame_roi = frame2[y-30:480, 0:x+w+30]
                            else:
                                frame_roi = frame2[y-30:y+h+30, 0:x+w+30]
                        elif x+w>610:
                            if y<30:
                                frame_roi = frame2[0:y+h+30, x-30:640]
                            elif y+h>450:
                                frame_roi = frame2[y-30:480, x-30:640]
                            else:
                                frame_roi = frame2[y-30:y+h+30, x-30:640]
                        else:
                            if y<30:
                                frame_roi = frame2[0:y+h+30, x-30:x+w+30]
                            elif y+h>450:
                                frame_roi = frame2[y-30:480, x-30:x+w+30]
                            else:
                                frame_roi = frame2[y-30:y+h+30, x-30:x+w+30]

                        
                        
                        if frame_roi.shape[0]<=10 or frame_roi.shape[1]<=10:
                            print("Shape", frame_roi.shape)
                            pass
                        else:
                            if change == 1:
                                print("Change Now")
                                while(now-now_p)<2:
                                    now = timer()
                                now_p = timer()
                                print("Change end")
                                change = 0
                            if now-now_p>0.2 and change!=1:
                                count+=1
                                flag = 1
                                cv2.imshow('frame_roi', frame_roi)
                                #filename = "~/Documents/IIT_M/%d.jpg"%p
                                #cv2.imwrite('filename',new)
                                #cv2.imshow('frame_roi_init', new)
                                
                                new = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
                                new = cv2.resize(new,(244,244), interpolation = cv2.INTER_LINEAR)
                                
                                #if cv2.waitKey(1) & 0xFF == ord('b'):
                                #    break
                                #new = np.expand_dims(new,axis=0)
                                pred = model.predict(new[np.newaxis, :, :, :])
                                #new = [new]
                                #t_imgs = np.array(new)
                                #print(new[0][0][0])

                                #t_imgs = np.append(t_imgs, new, axis=0)
                               
                                #model.trainable = True
                                #y_pred = model.predict(t_imgs,steps=1)
                                #print(full_pred[1])
                                #print(pred)
                                
                                #y_pred = full_pred[1]
                                
                                h=0
                                maxi=0
                                pos = 0
                                for h in range (27):
                                    if pred[0][h] > maxi:
                                        maxi = pred[0][h]
                                        pos = h
                                        #print(h)

                                labs_pred = pos #+1
                                count_pred.append(labs_pred)
                                
                                #print("  ")
                            else:
                                flag = 0

                            if p == 21:
                                pass
                            else:
                                if count>=10:
                                    final_pos = most_frequent(count_pred)
                                    beg = "A"
                                    if final_pos<20:
                                        beg = chr(ord(beg) + final_pos) 
                                    elif final_pos==20:
                                        beg = " "
                                        print("Space")
                                    else:
                                        beg = chr(ord(beg) + final_pos - 1)
                                    print("Predicted",beg)
                                    
                                    print("Change")

                                    count = 0
                                    count_pred = []
                                    change = 1

                                    word = word + beg
                    

    #cv2.imshow('frame', frame)


    p+=1
    cv2.putText(frame,word,(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame', frame)


    if flag == 1:    
        now_p = timer()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capa.release()
cv2.destroyAllWindows()