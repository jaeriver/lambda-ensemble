import json
import boto3
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

bucket_name = 'imagenet-sample'

s3_path = 'imagenet-sample-images/'
local_path = '/tmp/'

model_path = 'model/mobilenet_v2'

s3 = boto3.resource('s3')


# def download_image(object_path, file_path):
#     s3 = boto3.client('s3')
#     s3.download_file(bucket_name, object_path, file_path)

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
        # do something!!

        # download_image(file_list[i], local_path + file_name)
        img = read_image_from_s3(file_list[i])
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)

        # batchsize, 224, 224, 3
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        imgs.append(img)

    batch_imgs = np.vstack(imgs)
    return batch_imgs


def decode_predictions(preds, top=1):
    # get imagenet_class_index.json from container directory
    with open('imagenet_class_index.json') as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def inference_model(batch_imgs):
    model = load_model(model_path, compile=True)
    result = model.predict(batch_imgs)
    print(result)
    result = decode_predictions(result)
    results = []
    for single_result in result:
        single_result = [(img_class, label, str(round(acc * 100, 4)) + '%') for img_class, label, acc in single_result]
        results.append(single_result)
    return results


def lambda_handler(event, context):
    event = json.loads(event)
    file_list = event['file_list']
    batch_size = event['batchsize']

    batch_imgs = filenames_to_input(file_list, batch_size)
    result = inference_model(batch_imgs)
    return {
        'result': result
    }


s3 = boto3.resource('s3')
batchsize = 3
bucket = s3.Bucket(bucket_name)
filenames = [file.key for file in bucket.objects.all() if 'JPEG' in file.key]
event = json.dumps({'file_list': filenames, 'batchsize': batchsize})
context = 0
print(lambda_handler(event, context))
