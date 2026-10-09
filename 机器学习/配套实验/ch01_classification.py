"""实验1.2：固定分数的阈值与混淆矩阵。仅使用标准库。"""
labels = [1, 1, 0, 0, 1, 0, 0, 0]
scores = [0.9, 0.6, 0.8, 0.3, 0.4, 0.2, 0.55, 0.1]


def evaluate(labels, scores, threshold):
    if len(labels) != len(scores) or not labels:
        raise ValueError("inputs must have equal, nonzero lengths")
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = int(score >= threshold)
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return tp, fp, tn, fn, accuracy, precision, recall


for threshold in [0.5, 0.7]:
    result = evaluate(labels, scores, threshold)
    print("threshold:", threshold)
    print("TP FP TN FN:", *result[:4])
    print("accuracy={:.6f} precision={:.6f} recall={:.6f}".format(
        *result[4:]))
