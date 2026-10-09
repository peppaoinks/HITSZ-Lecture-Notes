"""实验1.1：固定划分上的均值基线与近邻回归。仅使用标准库。"""

train = [
    (0.5, 5.2), (1.0, 5.6), (1.5, 7.1), (2.0, 8.3),
    (2.5, 8.6), (3.0, 10.2), (3.5, 11.4), (4.0, 11.7),
    (4.5, 13.1), (5.0, 14.4), (5.5, 14.7), (6.0, 16.1),
]
valid = [(0.8, 5.7), (2.2, 8.2), (3.8, 11.8), (5.2, 14.3)]
test = [(1.2, 6.5), (2.8, 9.5), (4.2, 12.3), (5.8, 15.5)]


def mean_predict(data, x):
    return sum(y for _, y in data) / len(data)


def neighbor_predict(data, x, k):
    if not 1 <= k <= len(data):
        raise ValueError("k must be between 1 and the training size")
    ordered = sorted(data, key=lambda row: abs(row[0] - x))
    selected = ordered[:k]
    return sum(y for _, y in selected) / k


def mae(data, predict):
    errors = [abs(predict(x) - y) for x, y in data]
    return sum(errors) / len(errors)


print("split sizes:", len(train), len(valid), len(test))
print("baseline validation MAE: {:.6f}".format(
    mae(valid, lambda x: mean_predict(train, x))))
results = []
for k in [1, 3, 5]:
    score = mae(valid, lambda x: neighbor_predict(train, x, k))
    results.append((score, k))
    print("k={} validation MAE: {:.6f}".format(k, score))
_, best_k = min(results)  # 同分时选较小的k，规则预先确定
print("selected k:", best_k)
print("baseline test MAE: {:.6f}".format(
    mae(test, lambda x: mean_predict(train, x))))
print("selected model test MAE: {:.6f}".format(
    mae(test, lambda x: neighbor_predict(train, x, best_k))))
for x, y in test:
    prediction = neighbor_predict(train, x, best_k)
    print("x={:.1f} y={:.1f} prediction={:.3f} error={:.3f}".format(
        x, y, prediction, abs(prediction - y)))
