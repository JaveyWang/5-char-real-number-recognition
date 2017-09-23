# We know every chars in the image is always about the same size.
# And there are often only few grayscale values in the adhesions edge.
# Based on this, we implement the segment method as follow:
# 1.We find the characters inside the image by connected domain.
# 2.When two characters adhesions together, we will know it's wrong because the area
# of domain is too large.
# 3.We caculate the sum of grayscale values of every column in the connected domain rectangle.
# and we find the location of the min one which is the edge of adhesions.
import logging
import cv2
import numpy as np
import os

class Image(object):
    def __init__(self, img, idx_save=0):
        self.img = img
        self._img_paint = self.img.copy()
        self.list_crop_loc = []
        self.list_crop_img = []
        self._loc = (0, 0)
        self._idx_save = idx_save
        self._color_idx = 0

    def _ret_min_wh(self, img):
        img = np.squeeze(img)
        min_w, min_h = np.min(img, 0)
        return min_w, min_h

    # list_crop_img is a list of cropped images, every cropped image is a 2D numpy array.
    def find_char(self, dir_save=None, img=None, loc=None):
        """find all"""
        # assert isinstance(loc, (tuple, list)) # (w, h)
        loc = self._loc if loc is None else loc
        img = np.pad(self.img, ((1, 1), (1, 1)), 'constant') if img is None else np.pad(img, ((1, 1), (1, 1)), 'constant')
        loc = loc[0]-1, loc[1]-1
        ret, thresh = cv2.threshold(img, 20, 255, 0)
        logging.debug("thresh.shape:h,w{:}".format(np.shape(thresh)))
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=self._ret_min_wh) # (sort contours from left to right, up to down)
        logging.debug("len(contours):{:}".format(len(contours)))
        for i, c in enumerate(contours):
            logging.debug("i:{:}".format(i))
            rect = cv2.boundingRect(c)
            w1_relative, h1_relative = rect[0], rect[1]
            w2_relative, h2_relative = w1_relative + rect[2]-1, h1_relative + rect[3]-1
            w1, h1 = loc[0] + w1_relative, loc[1] + h1_relative
            w2, h2 = loc[0] + w2_relative, loc[1] + h2_relative
            area = (w2 - w1 + 1) * (h2 - h1 + 1)
            self._color_idx += 1
            VALUE_COLOR = 40 * self._color_idx
            if area < 70:
                # cv2.drawContours(img, rect, i, (255, 0, 0), 1)
                # logging.debug("contours:{:}".format(c))
                img_crop = img[h1_relative:h2_relative + 1, w1_relative:w2_relative + 1]
                img_crop = np.pad(img_crop, ((1, 1), (1, 1)), 'constant')
                cv2.rectangle(self._img_paint, (w1 - 1, h1 - 1), (w2 + 1, h2 + 1), (VALUE_COLOR, VALUE_COLOR, 0), 1)
                logging.debug("rect:w1:{:},h1:{:} w2:{:},h2:{:}".format(w1, h1, w2, h2))
                logging.debug("area:{:}".format(area))
                if dir_save:
                    self._idx_save += 1
                    path_save = os.path.join(dir_save, str(self._idx_save) + ".jpg")
                    cv2.imwrite(path_save, img_crop)
                    logging.info("cropped img {:} has been saved in:{:}".format(self._idx_save, path_save))

                self.list_crop_img.append(img_crop)
                self.list_crop_loc.append([[w1, h1], [w2, h2]])
            else:
                offset = np.argmin(np.sum(img[h1_relative:h2_relative + 1, w1_relative:w2_relative + 1], 0))
                if offset == 0:
                    offset = np.argmin(np.sum(img[h1_relative:h2_relative + 1, w1_relative + 1:w2_relative + 1], 0)) + 1
                # logging.debug("img[h1:h2, w1:w2]:{:}".format(np.sum(img[h1:h2, w1:w2], 0)))
                w3_relative, h3_relative = w1_relative + offset, h1_relative
                w4_relative, h4_relative = w2_relative, h2_relative
                w2_relative, h2_relative = w1_relative + offset - 1, h2_relative
                w3, h3 = loc[0] + w3_relative, loc[1] + h3_relative
                logging.debug("offset:{:}".format(offset))
                # Visualization
                self.find_char(dir_save, img[h1_relative:h2_relative+1, w1_relative:w2_relative+1], (w1, h1))
                self.find_char(dir_save, img[h3_relative:h4_relative+1, w3_relative:w4_relative+1], (w3, h3))
        return self.list_crop_img, self.list_crop_loc

    def get_paint(self):
        return self._img_paint