  
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import json
import argparse
import numpy as np
import json

def get_class_names(json_file="label_map.json"):
    with open(json_file, 'r') as f:
        class_names = json.load(f) 
    return class_names


def load_model(model_path="./model.h5"):
    model = tf.keras.models.load_model(model_path,
    custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def get_processed_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    return process_image(image)

def process_image(img, size=224):
    image = np.squeeze(img)
    image = tf.image.resize(image, (size, size))/255
    return image

def predict(image_path, model_path, top_k, category_names):
    image = get_processed_image(image_path)
    model = load_model(model_path)
    prediction = model.predict(np.expand_dims(image, axis=0))
    values, indices = tf.math.top_k(prediction, top_k)
    
    class_names = get_class_names(category_names)
    
    classes=[class_name[str(i)] for i in indices]
    
    print("values: {} and classes: {}".format(values,classes))
    return values.tonumpy()[0],classes
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',help="Path to image",default='./test_images/wild_pansy.jpg')
    parser.add_argument('model',help="Path to model", default='./model.h5')
    parser.add_argument('--top_k', help="top k predictions", default=5)
    parser.add_argument('--category_names',default='label_map.json',help="label map json file")
    args = parser.parse_args()
    
    predict(args.image_path, args.model, args.top_k, args.category_names)
    
    
    
    
    