import json
import boto3
import numpy as np
import time

table_name = 'lambda-ensemble1'
region_name = 'us-west-2'
dynamodb = boto3.resource('dynamodb', region_name=region_name)
table = dynamodb.Table(table_name)


def get_dynamodb(data):
    count = 0
    response = []
    print(data)
    for d in data:
        res = table.get_item(Key={"model_name": d['model_name'], "case_num": d['case_num']})
        response.append(list(res['Item'].values()))
        print(response)
    response = np.array(response)
    response = response.astype(np.float)
    response = response.sum(axis=0)
    response = response / len(data)
    return response


def decode_predictions(preds, top=1):
    # get imagenet_class_index.json from container directory
    with open('/var/task/lambda-ensemble/imagenet_class_index.json') as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def lambda_handler(event, context):
    model_num = len(event)
    batch_size = len(event[0]["batch_size"])

    result = get_dynamodb(event)

    return True
    similarity_list = []
    for single_result in result:
        single_result = [(img_class, label, round(acc * 100, 4)) for img_class, label, acc in single_result]
        results += single_result
    for img_idx in range(batch_size):
        acc = 0
        for model_idx in range(model_num):
            acc += event[model_idx]["result"][img_idx][2]
        acc /= model_num
        res = []
        res.append(event[0]["result"][img_idx][0])
        res.append(event[0]["result"][img_idx][1])
        res.append(acc)

        result.append(res)

    return {
        'result': result
    }


event = [{
    'model_name': "mobilenet_v2",
    'case_num': "1634866998.909735",
    'batch_size': '3',
    'total_time': 1,
    'pred_time': 1,
}]

context = 0
lambda_handler(event, context)
