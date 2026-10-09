"""第二章编程习题参考答案与核对。仅使用标准库，运行全部核对。"""
import math


def grade(score):
    if not 0 <= score <= 100:
        return None
    if score >= 90:
        return "A"
    if score >= 60:
        return "B"
    return "C"


def integer_sum(n):
    if n < 0:
        raise ValueError("n must be nonnegative")
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


def classify(score, threshold=60):
    if not 0 <= score <= 100:
        return None
    if score >= threshold:
        return "pass"
    return "fail"


def check_pairs(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("equal nonempty inputs required")


def mae(actual, predicted):
    check_pairs(actual, predicted)
    return sum(abs(p - y) for y, p in zip(actual, predicted)) / len(actual)


def mse(actual, predicted):
    check_pairs(actual, predicted)
    return sum((p - y) ** 2 for y, p in zip(actual, predicted)) / len(actual)


def clean_scores(raw):
    clean = []
    rejected = []
    for index, text in enumerate(raw):
        stripped = text.strip()
        if not (stripped.isascii() and stripped.isdecimal()):
            rejected.append((index, text, "invalid text"))
            continue
        value = int(stripped)
        if not 0 <= value <= 100:
            rejected.append((index, text, "out of range"))
            continue
        clean.append(value)
    return clean, rejected


def nearest_one(train, x):
    if not train:
        raise ValueError("training data are empty")
    ordered = sorted(train, key=lambda row: abs(row[0] - x))
    return ordered[0][1]


raw = " 82,91,0 "
values = [int(part.strip()) for part in raw.split(",")]
assert values == [82, 91, 0]
print("2.5:", values)

boundary = [-1, 0, 59, 60, 89, 90, 100, 101]
grades = [grade(v) for v in boundary]
assert grades == [None, "C", "C", "B", "B", "A", "A", None]
print("2.6:", grades)

v = list(range(7))
assert v[1:6:2] == [1, 3, 5]
assert v[-3:] == [4, 5, 6]
assert v[::-1] == [6, 5, 4, 3, 2, 1, 0]
assert v[4:2] == []
print("2.7:", v[1:6:2], v[-3:], v[::-1], v[4:2])

a = [1, 2]
b = [1, 2]
assert a.append([3, 4]) is None
assert b.extend([3, 4]) is None
assert a == [1, 2, [3, 4]] and b == [1, 2, 3, 4]
print("2.8:", a, b)

sums = [integer_sum(n) for n in [0, 1, 5]]
assert sums == [0, 1, 15]
print("2.9:", sums)

x = 1
steps = 0
while x < 20:
    x = 2 * x + 1
    steps += 1
assert (x, steps) == (31, 4)
print("2.10:", x, steps)

labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
counts = {}
for label in labels:
    counts[label] = counts.get(label, 0) + 1
assert counts == {"cat": 3, "dog": 2, "bird": 1}
print("2.11:", counts, sorted(set(labels)))

assert classify(70) == "pass"
assert classify(70, threshold=80) == "fail"
assert classify(-1) is None
print("2.12:", classify(70), classify(70, threshold=80))

actual = [1, 3, 5]
predicted = [2, 1, 5]
old_actual = actual[:]
old_predicted = predicted[:]
assert mae(actual, predicted) == 1
assert math.isclose(mse(actual, predicted), 5 / 3)
assert actual == old_actual and predicted == old_predicted
for actual_bad, predicted_bad in [([], []), ([1], [1, 2])]:
    for metric in [mae, mse]:
        try:
            metric(actual_bad, predicted_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("metric must reject invalid pairs")
print("2.13:", mae(actual, predicted), mse(actual, predicted))

values = [-2, 0, 3, -1, 4]
squares = [v ** 2 for v in values if v >= 0]
marks = ["valid" if v >= 0 else "invalid" for v in values]
assert squares == [0, 9, 16] and len(marks) == 5
print("2.14:", squares, marks)

raw = ["82", "", "0"]
original = raw[:]
values, rejected = clean_scores(raw)
assert values == [82, 0] and rejected == [(1, "", "invalid text")]
assert raw == original
print("2.15:", len(values), sum(values) / len(values))
empty, rejected_empty = clean_scores(["", "absent"])
assert empty == [] and len(rejected_empty) == 2
print("2.15 empty: no valid scores")

train = [(1, 5), (2, 7), (4, 11)]
evaluation = [(1.2, 5.5), (3.5, 10)]
predictions = [nearest_one(train, x) for x, _ in evaluation]
assert predictions == [5, 11]
assert mae([y for _, y in evaluation], predictions) == 0.75
# 距离同分时保留原记录顺序。
assert nearest_one([(1, 5), (3, 9)], 2) == 5
try:
    nearest_one([], 1)
except ValueError:
    pass
else:
    raise AssertionError("empty training set must be rejected")
print("2.16:", predictions, mae([y for _, y in evaluation], predictions))
print("all checks passed")
