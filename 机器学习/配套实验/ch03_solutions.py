"""Executable checks for selected Chapter 3 exercises."""
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory


def parse_nonnegative(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("not a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("expected finite nonnegative number")
    return value


def main():
    a = [[1, 2], [3, 4]]
    b, c, d = a, a.copy(), copy.deepcopy(a)
    b.append([5, 6])
    c[0][0] = 9
    assert a == [[9, 2], [3, 4], [5, 6]]
    assert c == [[9, 2], [3, 4]] and d == [[1, 2], [3, 4]]
    functions = [lambda x, k=k: x + k for k in range(3)]
    assert [f(10) for f in functions] == [10, 11, 12]
    assert parse_nonnegative(" 0 ") == 0.0
    for text in ["missing", "nan", "inf", "-1"]:
        try:
            parse_nonnegative(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"not rejected: {text}")
    with TemporaryDirectory() as folder:
        path = Path(folder) / "text.txt"
        path.write_text("重量\n费用\n", encoding="utf-8")
        assert path.read_text(encoding="utf-8").splitlines() == ["重量", "费用"]
    print("all checks passed")


if __name__ == "__main__":
    main()
