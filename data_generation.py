import numpy as np
import cv2

cat_img = cv2.imread('Data/Cat.jpg')
background = np.ones((256,256,3)) * 255

cat_h, cat_w, _ = cat_img.shape
sample = 1000
output = np.zeros([sample, 4])

for i in range(sample):
    init_x = np.random.randint(256-cat_w)
    init_y = np.random.randint(256-cat_h)

    render = background.copy()
    x1 = init_x
    y1 = init_y
    x2 = init_x+cat_w
    y2 = init_y+cat_h
    render[y1:y2, x1:x2, :] = cat_img
    cv2.imwrite('Data/img/' + str(i) + '.jpg', render)
    render = cv2.rectangle(render, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)
    cv2.imshow('',render)
    output[i,:] = [x1,y1,x2,y2]
    print([x1,y1,x2,y2],output[i])
    cv2.waitKey(1)

np.save('Data/target.npy', output)
print('Done')