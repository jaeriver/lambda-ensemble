import json
import boto3
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import (mobilenet_v2, nasnet, efficientnet)
from tensorflow.keras.models import load_model, save_model
import tensorflow.compat.v1.keras as keras

model_type = 'mobilenet_v2'
saved_model_dir = f'model/{model_type}'

# model = mobilenet_v2.MobileNetV2(weights='imagenet')
# temp = tf.zeros([8, 224, 224, 3])
# _ = mobilenet_v2.preprocess_input(temp)
# model.save(saved_model_dir)
# model_type = 'nasnetmobile'
# saved_model_dir = f'model/{model_type}'
#
# model = nasnet.NASNetMobile(weights='imagenet')
# model.save(saved_model_dir)


model_type = 'efficientnetb1'
saved_model_dir = f'model/{model_type}'

model = efficientnet.EfficientNetB1(weights='imagenet')
model.save(saved_model_dir)