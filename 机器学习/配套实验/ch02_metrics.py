"""实验2.2：从手算到函数的预测误差分析。仅使用标准库。"""


def validate_pairs(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("lengths differ")
    if not actual:
        raise ValueError("inputs are empty")


def mean_absolute_error(actual, predicted):
    validate_pairs(actual, predicted)
    total = 0.0
    for y, y_hat in zip(actual, predicted):
        total += abs(y_hat - y)
    return total / len(actual)


def mean_squared_error(actual, predicted):
    validate_pairs(actual, predicted)
    total = 0.0
    for y, y_hat in zip(actual, predicted):
        total += (y_hat - y) ** 2
    return total / len(actual)


actual = [6.0, 8.0, 10.0, 12.0]
predictions = {
    "A": [5.0, 9.0, 10.0, 14.0],
    "B": [6.0, 8.0, 10.0, 16.0],
}
for name, predicted in predictions.items():
    errors = [p - y for y, p in zip(actual, predicted)]
    absolute = [abs(error) for error in errors]
    worst_index = absolute.index(max(absolute))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    print("model:", name)
    print("errors:", errors)
    print("MAE={:.6f} MSE={:.6f} RMSE={:.6f}".format(mae, mse, mse ** 0.5))
    print("worst index:", worst_index)

# 有效值与边界条件均核对；错误路径应主动拒绝输入。
assert mean_absolute_error([2.0], [2.0]) == 0.0
assert mean_squared_error([2.0], [5.0]) == 9.0
for bad_actual, bad_predicted in [([], []), ([1.0], [1.0, 2.0])]:
    try:
        mean_absolute_error(bad_actual, bad_predicted)
    except ValueError:
        print("invalid input rejected:", bad_actual, bad_predicted)
    else:
        raise AssertionError("invalid input was accepted")
