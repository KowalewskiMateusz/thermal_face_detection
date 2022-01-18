from pathlib import Path
from glob import glob 
import pandas as pd
import ast 
import os
import random

from PIL import Image
from plantcv import plantcv as pcv
import numpy as np


IMG_HEIGHT=48
IMG_WIDTH=48
INPUT_SHAPE = ((IMG_HEIGHT,IMG_WIDTH,3))

def format_annotations(path):
    path = Path(path)
    paths = glob(str(path / '*.csv'))

    df = pd.DataFrame()
    for path in paths:
        path = Path(path)
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

def crop_and_save_negs(face_annotations):
    def random_crop(img):
        x = random.randint(0, img.shape[1] - IMG_WIDTH)
        y = random.randint(0, img.shape[0] - IMG_HEIGHT)
        img = img[y:y+IMG_HEIGHT, x:x+IMG_WIDTH]
        return img
    
    grouped = face_annotations.groupby('filename')
    j = 0

    for g_name in grouped.groups:
        g = grouped.get_group(g_name)
        movie = np.load(Path(f"numpy/a320/{g.iloc[0,].filename}.npy"))

        for _, row in g.iterrows():
            frame = movie[int(row.frame_number)]
            frame = pcv.transform.rescale(gray_img=frame)
            
            for _ in range(5):
                img = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
                img = random_crop(img)
                face_path = Path('datasets/eti_negs')
                morda = Image.fromarray(img)
                os.makedirs(face_path, exist_ok=True)
                morda.save(face_path / f"{j}.png")
                j += 1



face_annotations = format_annotations('annotations')
crop_and_save_negs(face_annotations)