# This expriment is for solve characters adhesions problem.
# 1.Resize the image to larger one by interpolation.
# 2.Define a threshold to filter the edge of character which do not adhesions anymore.
# 3.Now we can find all the characters inside the image by connected domain.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG)

DIR_DATASET = "../dataset/image"
PATH_IMG = os.path.join(DIR_DATASET, '12.png')
img = np.array(Image.open(PATH_IMG))
plt.subplot(411)
plt.imshow(img, cmap='gray')

img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
logging.debug("img.shape:{:}".format(np.shape(img)))
plt.subplot(412)
plt.imshow(img, cmap='gray')

ret, thresh = cv2.threshold(img, 50, 255, 0)
logging.debug("thresh.shape:{:}".format(np.shape(thresh)))
plt.subplot(413)
plt.imshow(thresh, cmap='gray')

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
logging.debug("len(contours):{:}".format(len(contours)))
logging.debug("contours:{:}".format(contours))
for i, c in enumerate(contours):
    # area = cv2.contourArea(c)
    # if area > 0:
    # cv2.drawContours(img, rect, i, (255, 0, 0), 1)
    rect = cv2.boundingRect(c)
    # logging.debug("contours:{:}".format(c))
    logging.debug("rect:{:}".format(rect))
    x1, y1 = rect[0], rect[1]
    x2, y2 = rect[0]+rect[2],rect[1]+rect[3]
    cv2.rectangle(img, (x1-1, y1-1), (x2, y2), (40*(i+1),0,0), 1)

plt.subplot(414)
plt.imshow(img, cmap='gray')
plt.show()
