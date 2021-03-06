import json
import boto3
import numpy as np
import time
import ast

table_name = 'lambda-ensemble1'
bucket_ensemble = 'lambda-ensemble'
s3 = boto3.resource('s3')
region_name = 'us-west-2'
dynamodb = boto3.resource('dynamodb', region_name=region_name)
table = dynamodb.Table(table_name)


def get_s3(data):
    bucket = s3.Bucket(bucket_ensemble)
    response = []
    for d in data:
        filename = d['model_name'] + '_' + d['case_num'] + '.txt'
        object = bucket.Object(filename)
        res = object.get()
        res = json.load(res['Body'])
        res = list(res.values())
        res = [ast.literal_eval(val) for val in res]
        response.append(res)
    response = np.array(response)
    response = response.astype(np.float)
    response = response.sum(axis=0)
    response = response / len(data)
    return response


def get_dynamodb(data):
    response = []
    for d in data:
        res = table.get_item(Key={"model_name": d['model_name'], "case_num": d['case_num']})
        res = res['Item']
        res.pop('model_name')
        res.pop('case_num')
        res = list(res.values())
        res = [ast.literal_eval(val) for val in res]
        response.append(res)
    response = np.array(response)
    response = response.astype(np.float)
    response = response.sum(axis=0)
    response = response / len(data)
    return response


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


def lambda_handler(event, context):
    results = []
    result = get_s3(event)
    result = decode_predictions(result)
    for single_result in result:
        single_result = [(img_class, label, round(acc * 100, 4)) for img_class, label, acc in single_result]
        results += single_result

    return {
        'result': results
    }


event = [{
    'model_name': "mobilenet_v2",
    'case_num': "1635068874.97596",
    'batch_size': '3',
    'total_time': 1,
    'pred_time': 1,
}, {
    'model_name': "mobilenet_v2",
    'case_num': "1635068874.97596",
    'batch_size': '3',
    'total_time': 1,
    'pred_time': 1,
}]

context = 0
lambda_handler(event, context)
