# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('a.jpg')   # you can read in images with opencv
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# sensitivity  = 100
# hsv_color1 = np.asarray((36, 25, 25))   # white!
# hsv_color2 = np.asarray([70, 255,255])
#    # yellow! note the order

# mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

# plt.imshow(mask)   # this colormap will display in black / white
# plt.show()

import cv2
import numpy as np

## Read
img = cv2.imread("OpenMap.jpg")

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
white = 0
black = 0
print(hsv.shape)
white = np.sum(hsv == 255)  
black = np.sum(hsv == 0) 
print(white, black, white+black)
print("percent: " + str(black/(white+black)*100))