import json
import boto3
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import time

bucket_name = 'imagenet-sample'
model_path = '/var/task/lambda-ensemble/model/mobilenet_v2'
model = load_model(model_path, compile=True)

s3 = boto3.resource('s3')


def read_image_from_s3(filename):
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(filename)
    response = object.get()
    file_stream = response['Body']
    img = Image.open(file_stream)
    return img


def filenames_to_input(file_list, batchsize):
    imgs = []
    for i in range(batchsize):
        img = read_image_from_s3(file_list[i])
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)

        # batchsize, 224, 224, 3
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        imgs.append(img)

    batch_imgs = np.vstack(imgs)
    return batch_imgs


def inference_model(batch_imgs):
    pred_start = time.time()
    result = model.predict(batch_imgs)
    pred_time = time.time() - pred_start

    return json.dumps(result.tolist()), pred_time


def lambda_handler(event, context):
    file_list = event['file_list']
    batch_size = event['batchsize']

    batch_imgs = filenames_to_input(file_list, batch_size)
    total_start = time.time()
    result, pred_time = inference_model(batch_imgs)
    total_time = time.time() - total_start
    return {
        'result': result,
        'total_time': total_time,
        'pred_time': pred_time,
    }
