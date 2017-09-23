# This file is for create .csv labels.
import logging
import os
import csv

logging.basicConfig(level=logging.INFO)

DIR_DATASET = "../dataset"
PATH_LABELS = os.path.join(DIR_DATASET, "label.txt")
PATH_SEGM_LABELS = os.path.join(DIR_DATASET, "segm_labels.csv")
with open(PATH_LABELS, "r") as f:
    if os.path.exists(PATH_SEGM_LABELS):
        with open(PATH_SEGM_LABELS, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            for labels in csv_reader:
                logging.info("len(labels):{:}".format(len(labels)))
                logging.info("labels:{:}".format(labels))
        logging.info("Already exist segm_labels.csv")
    else:
        labels = list([char for line in f.readlines() for char in line.strip("\n")])
        for idx, label in enumerate(labels):
            if label == ".":
                labels[idx] = str(10)
            else:
                labels[idx] = label
        logging.info("len(labels):{:}".format(len(labels)))
        logging.info("labels:{:}".format(labels))
        with open(PATH_SEGM_LABELS, "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            for label in labels:
                csv_writer.writerow([label])
        logging.info("segment labels have been writed to csvfile which in {:}".format(PATH_SEGM_LABELS))

with open(PATH_SEGM_LABELS, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for label in csv_reader:
        logging.info(label)