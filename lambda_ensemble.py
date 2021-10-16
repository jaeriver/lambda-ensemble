import json


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
    batch_size = len(event[0]["result"])
    result = []

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
