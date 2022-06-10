from tensorflow.keras.applications.mobilenet import preprocess_input
from drawing.hot_encoder import one_hot_encoder
from drawing.datagen import draw_image_cv2
from drawing import csv_paths
import pandas as pd
import numpy as np
import json
import gc


def create_validation_set(wordEncoder, class_num, img_size, thickness=6, time_color=True):
  
    x = np.zeros((80000, img_size, img_size, 1))
    y = np.zeros((80000, class_num, 1))

    df = pd.read_csv(csv_paths[-1]) 
    df['drawing'] = df['drawing'].apply(json.loads)
    print(csv_paths[-1])
    for k, raw_strokes in enumerate(df['drawing']):
        x[k,:,:,0] = draw_image_cv2(raw_strokes, img_size, thickness=thickness, time_color=time_color)
        y[k,:,:] = one_hot_encoder(df['word'].iloc[k], wordEncoder)
    x = preprocess_input(x).astype(np.float32)
    del(df)
    gc.collect()
    return x, y