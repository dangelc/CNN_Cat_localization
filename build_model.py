import numpy as np
import cv2
import os

# Reading data
Y = np.load('Data/target.npy')
X = os.path.exists('Data/images.npy')

if not X:
    X = np.zeros((1, 256, 256))

    path = 'Data/img/'
    imgs = os.listdir(path)

    for img in range(1000):
        temp = cv2.imread(path+str(img)+'.jpg', 0)
        X = np.vstack((X, 
                    temp.reshape((1, temp.shape[0], temp.shape[1]))))
        
    X = X[1:]
    X = X.reshape((X.shape[0], 256, 256, 1))
    np.save('Data/images.npy', X)
else:
    X = np.load('Data/images.npy')
print(X.shape, Y.shape)
Y = Y / 256

# Create model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Input

class model_v1(Sequential):
    def __init__(self):
        super().__init__()
        # Block1
        self.add(Input((256, 256, 1)))
        self.add(Conv2D(16, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block2
        self.add(Conv2D(32, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block3
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block4
        self.add(Conv2D(128, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block5
        self.add(Conv2D(256, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block6
        self.add(Flatten())
        self.add(Dense(256, activation='relu'))
        self.add(Dense(4))

        self.compile(optimizer='adam',
                    loss = 'mse',
                    metrics=['mse'])
        
class model_v2(Sequential):
    def __init__(self):
        super().__init__()
        # Block1
        self.add(Input((256,256,1)))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block2
        self.add(Conv2D(128, (3,3), activation='relu'))
        self.add(Conv2D(128, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block3
        self.add(Conv2D(256, (3,3), activation='relu'))
        self.add(Conv2D(256, (3,3), activation='relu'))
        self.add(Conv2D(256, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block4
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block5
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(Conv2D(512, (3,3), activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPool2D(2))
        # Block6
        self.add(Flatten())
        self.add(Dense(256, activation='relu'))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(4))

        self.compile(optimizer='adam',
                    loss = 'mse',
                    metrics=['mse'])
        

model = model_v2()
print(model.summary())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

history = model.fit(x_train, y_train, epochs=18, steps_per_epoch=100)
print(model.evaluate(x_test, y_test, batch_size=8, verbose=2))
model.save(filepath='model/cat_v2.tf', save_format='tf')

n = x_test.shape[0]
for i in range(n):
    img = x_test[i].reshape((256,256))
    print(x_test[i].shape)
    coors_pred = model.predict(x_test[i].reshape(1,256,256,1))[0]
    coors_pred = coors_pred * 256
    print(coors_pred)
    x1, y1, x2, y2 = coors_pred
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    img = cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2,y2), color=(0), thickness=2)
    cv2.imshow('pred', img)
    k = cv2.waitKey(1000) & 0xFF
    if k == ord('q'):
        break