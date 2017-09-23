# This file is for create .pkl datasets file.
import os
import logging
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

DIR_IMG_ORIG = "../dataset/train"
DIR_DATASET = "../dataset"
FILENAME_PKL = "train.pkl"

folder = os.listdir(DIR_IMG_ORIG)
folder.sort(key=lambda x:int(x.split(".")[0]))
logging.info("{:}".format(folder))

imgs = []
for file in folder:
    if file.endswith("jpg"):
        img = Image.open(os.path.join(DIR_IMG_ORIG, file))
        # The width and height of every image inside list imgs is diffrent, it can not be transformed to array as a whole
        arr = np.array(img, np.float32)
        arr = arr[np.newaxis,:,:,np.newaxis]
        logging.debug("arr.shape:{:}".format(arr.shape))
        imgs.append(arr)
logging.info("len(imgs):{:}".format(len(imgs)))

path_pkl = os.path.join(DIR_DATASET, FILENAME_PKL)
if os.path.exists(path_pkl):
    logging.info("Already exist {:} file.".format(FILENAME_PKL))
else:
    with open(path_pkl, "wb") as f:
        pickle.dump(imgs, f)
        logging.info("Dump {:} file succesful.".format(FILENAME_PKL))

with open(path_pkl, "rb") as f:
    imgs_load = pickle.load(f)
    logging.info(len(imgs_load))
