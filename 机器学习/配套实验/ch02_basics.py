"""实验2.1：可追踪的数据清理与统计。仅使用标准库。"""
raw = [" 82 ", "91", "", " 76", "absent", "88", "0", "105", "-3"]
clean = []
rejected = []
for index, text in enumerate(raw):
    stripped = text.strip()
    digits = stripped.isascii() and stripped.isdecimal()
    if not digits:
        rejected.append((index, text, "not an unsigned ASCII integer"))
        continue
    value = int(stripped)
    if not 0 <= value <= 100:
        rejected.append((index, text, "out of range"))
        continue
    clean.append(value)


def describe(values):
    if not values:
        return None
    count = len(values)
    average = sum(values) / count
    passed = sum(value >= 60 for value in values)
    return {
        "count": count,
        "mean": average,
        "minimum": min(values),
        "maximum": max(values),
        "pass_rate": passed / count,
    }


print("raw:", raw)
print("clean:", clean)
print("rejected:")
for row in rejected:
    print(row)
summary = describe(clean)
print("summary:")
for key, value in summary.items():
    print(key, "=", value)
print("empty input:", describe([]))
