import logging
import os
import pathlib
from typing import overload
import xml.etree.ElementTree as ET
import pandas as pd 
from glob import glob
import cv2
import ast 
from plantcv import plantcv as pcv
import numpy as np
import random


def format_annotations(path):
    path = pathlib.Path(path)
    paths = glob(str(path / '*.csv'))

    df = pd.DataFrame()
    for path in paths:
        path = pathlib.Path(path)
        data = pd.read_csv(path)
        data = data[data.region_shape_attributes.str.len() > 2]

        if data.shape[0] == 0:
            continue

        data.region_shape_attributes = data.apply(lambda x: ast.literal_eval(x.region_shape_attributes), axis=1)
        annotations_info = data.apply(lambda x: [x.region_shape_attributes["x"], x.region_shape_attributes["y"],
                                                 x.region_shape_attributes["width"],
                                                 x.region_shape_attributes["height"]], axis=1, result_type='expand')

        annotations_info.columns = ['x', 'y', 'width', 'height']
        data = data.merge(annotations_info, left_index=True, right_index=True)
        data.drop(columns=['file_size', 'file_attributes', 'region_shape_attributes', 'region_attributes'],
                  inplace=True)

        data.filename = data.apply(lambda x: str(x.filename)[:-4], axis=1)
        data.rename(columns={'filename': 'frame_number'}, inplace=True)

        data.insert(0, 'filename', path.parts[-1][:-8])

        df = df.append(data)

    return df.reset_index(drop=True)



class VOCDataset:

    def __init__(self, root,  transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.class_dict = {'BACKGROUND' : 0  ,'FACE' :1}
        self.is_test = is_test
        if self.is_test:
            self.annotations = 'data/annotations/test'
        else:
            self.annotations = 'data/annotations/train' 
        
        self.annotations = format_annotations(self.annotations)
        self.ids = dict()
        for i, row in self.annotations.iterrows():
            self.ids[i] = f'{row["filename"]}/{row["frame_number"]}' 
        
    def __getitem__(self, index):
        
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids
    
    @staticmethod
    def check_intersect(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea > 0

    @staticmethod
    def random_crop(boxes):
        negs = list()
        while len(negs)< 15:
            overlap = False
            width  = random.randint(30,70)
            height = width + random.randint(-5,5)
            x = random.randint(0, 320 - width)
            y = random.randint(0, 240 - height)
            for _,box in boxes.iterrows():
                if VOCDataset.check_intersect((box['x'],box['y'],box['x'] + box['width'],box['y'] + box['height']),(x,y,x+width,y+height)):
                    overlap = True
            if not overlap:
                negs.append({'x':x,'y':y,'width':width,'height':height})
        return negs

    def _get_annotation(self, image_id):
        filename,frame_number = image_id.split("/")
        ann = self.annotations.loc[(self.annotations['filename'] ==filename) & (self.annotations['frame_number'] ==frame_number)  ] 
        
        boxes = []
        labels = []
        for _,row in ann.iterrows():
                x1 = float(row['x']) 
                y1 = float(row['y']) 
                x2 = float(row['x'] + row['width']) 
                y2 = float(row['y'] + row['height']) 
                boxes.append([x1, y1, x2, y2])
            
                labels.append(self.class_dict['FACE'])

        if not self.is_test:
            negs = self.random_crop(ann)
            for row in negs:
                x1 = float(row['x']) +float(row['x'])
                y1 = float(row['y']) +float(row['y'])
                x2 = float(x1 + row['width']) 
                y2 = float(y1 + row['height'])
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict['BACKGROUND'])


        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        image_file = self.root / f"{image_id}.png"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = pcv.transform.rescale(gray_img=image)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        return image
