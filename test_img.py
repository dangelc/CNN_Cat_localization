import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

def draw_cat(event,x,y,flags,param):
    global background, cat, render
    if event == cv2.EVENT_LBUTTONDOWN:
        render = background.copy()
        x1 = int(x - cat.shape[1]/2)
        y1 = int(y - cat.shape[0]/2)
        x2 = int(x + cat.shape[1]/2)
        y2 = int(y + cat.shape[0]/2)
        render[y1:y2, x1:x2, :] = cat

model = load_model('model/cat_v2.tf')

scr1 = 'drawing'
scr2 = 'AI found'

cat = cv2.imread('Data/Cat.jpg')
background = np.ones((256,256,3)) * 255
background = background.astype('float32')
render = background.copy()
cv2.imshow(scr1,background)
cv2.setMouseCallback(scr1,draw_cat)

while True:
    prediction = render.copy()
    input = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
    coors_pred = model.predict(input.reshape(1,256,256,1))[0]
    coors_pred = coors_pred * 256
    print(coors_pred)
    x1, y1, x2, y2 = coors_pred
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    prediction = cv2.rectangle(prediction, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=2)
    cv2.imshow(scr1, render)
    cv2.imshow(scr2, prediction)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break