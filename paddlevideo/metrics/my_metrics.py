import numpy as np
import paddle

def metrics_train(x, out, y, num_sample, num_top1):
    # Calculating Recognition Accuracies
    num_sample += x.shape[0]
    reco_top1 = paddle.argmax(out, 1)

    num_top1 += paddle.sum(paddle.equal(reco_top1, y)).item()
    return num_top1, num_sample


def metrics_eval(x, out, y, num_sample, num_top1, num_top5, cm):
    # Calculating Recognition Accuracies
    num_sample += x.shape[0]
    reco_top1 = paddle.argmax(out, 1)
    num_top1 += paddle.sum(paddle.equal(reco_top1, y)).item()
   
    reco_top5 = paddle.topk(out,5)[1]
    num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.shape[0])])


    # Calculating Confusion Matrix
    for i in range(x.shape[0]):
        cm[y[i], reco_top1[i]] += 1
    return num_top1, num_top5, num_sample, cm