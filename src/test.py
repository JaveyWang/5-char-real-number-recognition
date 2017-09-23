import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import src.util.segment as segment
from src.util.util import intlist2str

def plot(list_img, pred_string):
    numb_img = len(list_img)
    for i, img in enumerate(list_img):
        plt.subplot(str(numb_img)+"1"+str(i+1))
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(pred_string, fontdict={"fontsize":10}, loc="left")

logging.basicConfig(level=logging.INFO)

DIR_DATASET = "../dataset"
DIR_IMG = os.path.join(DIR_DATASET, "image")
FILENAME_PKL = "train.pkl"
PATH_LABELS = os.path.join(DIR_DATASET, 'label.txt')
# Load all the image
FILENAME_IMG_LIST = os.listdir(DIR_IMG)
FILENAME_IMG_LIST.sort(key=lambda x:int(x.split(".")[0]))
logging.debug("len(FILENAME_IMG_LIST):{:}".format(len(FILENAME_IMG_LIST)))

# Load dataset images and labels
imgs = []
for file_name in FILENAME_IMG_LIST:
    if file_name.endswith("png"):
        path_img = os.path.join(DIR_IMG, file_name)
        logging.debug("path_img:{:}".format(path_img))
        img = np.array(Image.open(path_img))
        imgs.append(img)

with open(PATH_LABELS, 'r') as f:
    labels_str = np.array([line.strip('\n') for line in f.readlines()])
    logging.info('labels.shape:{:} labels:{:}'.format(labels_str.shape, labels_str[0]))


# img_idx = 1
# PATH_IMG = os.path.join(DIR_IMG, str(img_idx) +".png")
# img = np.array(Image.open(PATH_IMG))
# plt.imshow(img, cmap='gray')
# plt.title(labels_str[img_idx-1])
# plt.show()

DIR_CKPT = "../ckpt"
PATH_CKPT_META = os.path.join(DIR_CKPT, "model.ckpt-150.meta")
PATH_CKPT = os.path.join(DIR_CKPT, "model.ckpt-150")

tf.reset_default_graph()
loaded_graph = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=loaded_graph) as sess:
    ckpt = tf.train.get_checkpoint_state(DIR_CKPT)
    loader = tf.train.import_meta_graph(PATH_CKPT_META)
    loader.restore(sess, PATH_CKPT)
    logging.debug(ckpt.model_checkpoint_path)
    # Get tensor name
    x = loaded_graph.get_tensor_by_name("x:0")
    y = loaded_graph.get_tensor_by_name("y:0")
    logits = loaded_graph.get_tensor_by_name('logits/logits:0')

    for i, img in enumerate(imgs):
        img = segment.Image(img)
        list_crop_img, list_crop_loc = img.find_char()
        logits_str  = []
        for crop_img in list_crop_img:
            logits_char = sess.run(logits, feed_dict={x: crop_img[np.newaxis, :, :, np.newaxis]})
            logits_str.append(np.squeeze(logits_char))
        logits_str = np.array(logits_str)
        logging.debug("str_logit.shape:{:}".format(logits_str.shape))

        pred_string = intlist2str(np.argmax(logits_str, 1))
        logging.info("pred_string:{:}".format(pred_string))

        plot([img.img, img.get_paint(), img.img], pred_string)
        plt.show()
