"""Minimal meaningful checks: hand values, bad input, state and data alignment."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from io_utils import read_records
from metrics import mae
from models import MeanRegressor, NearestRegressor


class ProjectTests(unittest.TestCase):
    def test_hand_calculation(self):
        self.assertAlmostEqual(mae([5, 5], [7, 3]), 2.0)
        model = MeanRegressor().fit([1, 2, 3], [2, 2, 8])
        self.assertEqual(model.predict([99, -1]), [4.0, 4.0])
        with self.assertRaises(ValueError):
            MeanRegressor().fit([1, 2], [1e308, 1e308])

    def test_invalid_metrics(self):
        for y, p in [([], []), ([1], []), ([1], [float("nan")]),
                     ([-1e308], [1e308])]:
            with self.assertRaises(ValueError):
                mae(y, p)

    def test_model_state_and_copy(self):
        with self.assertRaises(RuntimeError):
            MeanRegressor().predict([1])
        x, y = [1, 3], [6, 10]
        model = NearestRegressor().fit(x, y)
        x[0], y[0] = 99, 99
        self.assertEqual(model.predict([2]), [6])  # first tie wins
        with self.assertRaises(ValueError):
            NearestRegressor().fit([-1e308], [1]).predict([1e308])

    def test_bad_csv(self):
        cases = [
            "id,split,weight,cost\na,train,nan,6\n",
            "id,split,weight,cost\na,train,1,6\na,test,2,8\n",
            "id,split,weight,cost\na,other,1,6\n",
            "id,split,weight,cost\na,train,1\n",
            "id,split,weight,cost\na,train,1,6,extra\n",
            "id,split,weight,cost\na,train,1,6\n",
        ]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for content in cases:
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_records(path)
            # A quoted field spans two physical lines; the next bad row is line 4.
            path.write_text('id,split,weight,cost\n"a\nb",train,1,6\n'
                            'b,validation,nan,8\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row 4:"):
                read_records(path)

    def test_dataset_values(self):
        path = Path(__file__).resolve().parent / "data" / "parcels.csv"
        rows = read_records(path)
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
        model = NearestRegressor().fit([r["weight"] for r in train],
                                       [r["cost"] for r in train])
        predicted = model.predict([r["weight"] for r in test])
        self.assertEqual(predicted, [6.0, 10.0, 10.0, 12.0])
        self.assertAlmostEqual(mae([r["cost"] for r in test], predicted), .85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
