from tensorflow.keras.applications.mobilenet import preprocess_input
from drawing.hyperparameter import BASE_IMG_SIZE
from drawing.hot_encoder import one_hot_encoder
from drawing import csv_paths
from random import choice
import pandas as pd
import numpy as np
import json
import cv2


def draw_image_cv2(strokes, img_size, thickness=6, time_color=True):
    img = np.zeros((BASE_IMG_SIZE, BASE_IMG_SIZE), np.uint8)
    for t, stroke in enumerate(strokes):
        for i in range(len(stroke[0])-1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]),color, thickness)
    
    if img_size != BASE_IMG_SIZE:
        return cv2.resize(img, (img_size, img_size))
    else: 
        return img

def training_data_generator(batch_size, img_size, num_class, wordEncoder, thickness=6, time_color=True):
    while True:
        x = np.zeros((batch_size, img_size, img_size, 1))
        y = np.zeros((batch_size, num_class, 1))
        df = pd.read_csv(choice(csv_paths[0:-1])).sample(n=batch_size)
        df['drawing'] = df['drawing'].apply(json.loads)
        for k, raw_strokes in enumerate(df['drawing']):
            x[k, :, :, 0] = draw_image_cv2(raw_strokes, img_size,thickness=6, time_color=False)
            y[k, :, :] = one_hot_encoder(df['word'].iloc[k], wordEncoder)
        x = preprocess_input(x).astype(np.float32)
        yield x, y

def df_to_images(df, img_size, thickness=6, time_color=True):
    x = np.zeros((len(df), img_size, img_size, 1))
    df['drawing'] = df['drawing'].apply(json.loads)
    for i , raw_strokes in enumerate(df['drawing']):
        x[i, :, :, 0] = draw_image_cv2(raw_strokes, img_size = img_size)
    x = preprocess_input(x).astype(np.float32)
    return x
