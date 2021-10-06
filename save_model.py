import json
import boto3
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.models import load_model, save_model
import tensorflow.compat.v1.keras as keras

model_type = 'mobilenet_v2'

saved_model_dir = f'model/{model_type}'

model = mobilenet_v2.MobileNetV2(weights='imagenet')
temp = tf.zeros([8, 224, 224, 3])
_ = mobilenet_v2.preprocess_input(temp)

# tf.keras.Model.save(session=keras.backend.get_session(),
#                            export_dir=saved_model_dir,
#                            inputs={'input_1:0': model.inputs[0]},
#                            outputs={'probs/Softmax:0': model.outputs[0]})
model.save(saved_model_dir)