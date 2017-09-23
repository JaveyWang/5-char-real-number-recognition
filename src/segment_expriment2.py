# This expriment is for solve characters adhesions problem by other method.
# see segment.Image for more details.
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import src.util.segment as segment

logging.basicConfig(level=logging.INFO)

DIR_DATASET = "../dataset/image"
PATH_IMG = os.path.join(DIR_DATASET, '14.png')
img_orig = np.array(Image.open(PATH_IMG))
plt.subplot(411)
plt.imshow(img_orig, cmap='gray')

img = segment.Image(img_orig)
list_crop_img, list_crop_loc = img.find_char("../dataset")
img_paint = img.get_paint()
# list_crop_img, list_crop_loc = find_char(img, (0, 0), img_paint)
logging.info("list_crop_loc:{:}".format(list_crop_loc))
logging.info("len(list_crop_img):{:}".format(len(list_crop_img)))
# logging.info("list_crop_img:{:}".format(list_crop_img))
# for crop_img in list_crop_img:
#     plt.subplot(412)
#     plt.imshow(crop_img, cmap="gray")
#     plt.show()
plt.subplot(413)
plt.imshow(img_paint, cmap='gray')
plt.show()