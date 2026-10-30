"""Regression tests for the evaluation output-directory contract.

The __main__ entry point creates the directory before Isaac Kit starts so that
--portable-root exists.  main() used to create/check it a second time and
always raised FileExistsError, leaving the BO controller without a summary.
"""
import ast
from pathlib import Path
import tempfile
import unittest

BO = Path(__file__).resolve().parents[1]


def load_function(filename, name, namespace):
    tree = ast.parse((BO / filename).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(BO / filename), "exec"), namespace)
    return namespace[name]


class EvaluationOutputDirTests(unittest.TestCase):
    def setUp(self):
        self.prepare = load_function(
            "bo_finetune_evaluate.py", "prepare_output_dir", {"Path": Path}
        )

    def test_precreated_output_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evaluation"
            output.mkdir()
            self.prepare(output, already_prepared=True)
            self.assertTrue(output.is_dir())

    def test_existing_output_is_rejected_without_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evaluation"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self.prepare(output)

    def test_missing_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "evaluation"
            self.prepare(output)
            self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main()
