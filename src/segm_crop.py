import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from PIL import Image
import src.util.segment as segment

logging.basicConfig(level=logging.INFO)

DIR_DATASET = "../dataset"
DIR_IMAGE = os.path.join(DIR_DATASET, "image")
DIR_SAVE = "../dataset/segm_crop"

FILENAME_IMG_LIST = os.listdir(DIR_DATASET)
FILENAME_IMG_LIST.sort(key=lambda x:int(x.split(".")[0]))
logging.debug("len(FILENAME_IMG_LIST):{:}".format(len(FILENAME_IMG_LIST)))

path_save = "../dataset/train"
idx = 0
for file_name in FILENAME_IMG_LIST:
    if file_name.endswith("png"):
        path_img = os.path.join(DIR_DATASET, file_name)
        logging.debug("path_img:{:}".format(path_img))
        img_orig = np.array(Image.open(path_img))
        img = segment.Image(img_orig, idx_save=idx)
        plt.subplot(311)
        plt.imshow(img_orig, cmap="gray")
        # path_save = os.path.join(DIR_SAVE, str(idx))
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # list_crop_img, list_crop_loc = img.find_char((0, 0), img_paint, "../dataset")
        list_crop_img, list_crop_loc = img.find_char(path_save)
        logging.info("len(list_crop_loc):{:}".format(len(list_crop_loc)))
        idx += len(list_crop_img)
        # img_paint = img.get_paint()
        # for crop_img in list_crop_img:
        #     plt.subplot(312)
        #     plt.imshow(crop_img, cmap="gray")
        #     plt.show()
        # plt.subplot(313)
        # plt.imshow(img_paint, cmap='gray')
        # plt.show()
logging.info("idx:{:}".format(idx))
