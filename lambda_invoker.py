import json

def lambda_handler(event, context):
    model_num = len(event)
    batch_size = len(event[0]["result"])
    result = []
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
