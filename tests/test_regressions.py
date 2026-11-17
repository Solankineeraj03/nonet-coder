"""Regression tests for milestone-1 fixes. Run: python3 -m unittest discover tests"""
import os, subprocess, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nonet import prior
from nonet.synth import doc_constants
from nonet.extract import extract_examples, classify_doc



class PriorShape(unittest.TestCase):
    def test_word_in_wc_but_not_in_cond(self):
        saved = prior._P
        prior._P = {"n": 10, "base": {"len": 0.25}, "cond": {}, "wc": {"comma": 3}}
        try:
            p = prior.prob("len", {"comma"}, False)
            self.assertEqual(p, 0.25)  # missing cond entry counts as zero, so the base-rate floor applies
        finally:
            prior._P = saved

    def test_real_prior_handles_all_wc_words(self):
        words = set(prior._P["wc"]) - set(prior._P["cond"])
        for op in ("len", "add"):
            prior.cost(op, words, False)


class ConstantProvenance(unittest.TestCase):
    def test_example_outputs_not_added(self):
        doc = "Return whether the list is balanced.\n>>> f([1, 2])\n'ok'\n>>> f([3])\n'bad'\n"
        nums, strs = doc_constants(doc, [([[1, 2]], "ok"), ([[3]], "bad")], "f")
        self.assertNotIn("ok", strs)
        self.assertNotIn("bad", strs)

    def test_prose_labels_kept(self):
        doc = "Return 'YES' if the list is balanced, otherwise 'NO'.\n>>> f([1, 2])\n'YES'\n"
        nums, strs = doc_constants(doc, [([[1, 2]], "YES")], "f")
        self.assertTrue({"YES", "NO"} <= strs)

    def test_example_numbers_not_added(self):
        doc = "Add 7 to x.\n>>> f(1)\n8\n"
        nums, _ = doc_constants(doc, [([1], 8)], "f")
        self.assertIn(7, nums)
        self.assertNotIn(8, nums)


def consts_for(prompt, entry):
    info, ex = extract_examples(prompt, entry)
    return doc_constants(info["doc"], ex, entry, info["ex_lines"])


def mk(entry, args, doc):
    body = "\n".join("    " + l if l else "" for l in doc.strip("\n").split("\n"))
    return f'def {entry}({args}):\n    """\n{body}\n    """\n'


class SpanProvenance(unittest.TestCase):
    def test_multiline_input_output_block(self):  # reproduced leak from review 2
        nums, strs = consts_for(mk("lab", "x", "Label x.\nExample:\nInput:\n  x: 918\nOutput:\n  'ok'\n"), "lab")
        self.assertNotIn(918, nums)
        self.assertNotIn("ok", strs)

    def test_single_line_input_output(self):
        nums, strs = consts_for(mk("lab", "x", "Label x.\nInput: 918\nOutput: 'ok'\n"), "lab")
        self.assertNotIn(918, nums)
        self.assertNotIn("ok", strs)

    def test_arrows(self):
        for arrow in ("==>", "=>", "->", "➞", "should return"):
            nums, strs = consts_for(mk("lab", "x", f"Label x.\nlab(918) {arrow} 'zz'\n"), "lab")
            self.assertNotIn(918, nums, arrow)
            self.assertNotIn("zz", strs, arrow)

    def test_doctest_quoted_and_multiline_result(self):
        doc = "Label x.\n>>> lab(918)\n'ok'\n>>> lab(77)\n[455,\n 456]\n"
        nums, strs = consts_for(mk("lab", "x", doc), "lab")
        self.assertFalse({918, 77, 455, 456} & nums)
        self.assertNotIn("ok", strs)

    def test_prose_after_example_is_kept(self):  # false exclusion from review 2
        doc = "Label x.\n>>> lab(918)\n'YES'\nReturn the label 'YES' when valid, else 'NO', using base 7 here\n"
        nums, strs = consts_for(mk("lab", "x", doc), "lab")
        self.assertTrue({"YES", "NO"} <= strs)
        self.assertIn(7, nums)
        self.assertNotIn(918, nums)

    def test_explanation_block_excluded(self):
        doc = "Sum the digits.\nExample:\n  lab(123) == 6\nExplanation:\n  1 + 2 + 3 = 6\n\nUse base 7 here\n"
        nums, _ = consts_for(mk("lab", "x", doc), "lab")
        self.assertEqual(nums, {7})

    def test_extractor_lines_are_never_prose(self):
        doc = "Label x.\nFor x = 918 the output should be 'ok'\n"
        info, ex = extract_examples(mk("lab", "x", doc), "lab")
        self.assertEqual(ex, [([918], "ok")])
        kinds = classify_doc(info["doc"], "lab", info["ex_lines"])
        self.assertEqual(kinds[1], "example")


class Scoring(unittest.TestCase):
    """Behavioral checks of the scoring path actually used by run_eval.work."""

    @classmethod
    def setUpClass(cls):
        cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        try:
            import run_eval
        finally:
            os.chdir(cwd)
        cls.re = run_eval

    def test_reasons(self):
        score = self.re.score
        self.assertEqual(score("def f(x):\n    return x+1\nassert f(1) == 2\nprint('noise')\n", "runner"), "pass")
        self.assertEqual(score("assert 1 == 2\n", "runner"), "assert")
        self.assertEqual(score("{}['k']\n", "runner"), "exception")
        self.assertEqual(score("def (:\n", "runner"), "runner-error")
        self.assertEqual(score("assert 1 == 2\n", "legacy"), "assert")

    def test_cpu_limit(self):
        self.assertEqual(self.re.score("while True:\n    pass\n", "runner", cpu=2), "cpu-limit")

    def test_work_scores_through_runner(self):
        calls = []
        real_run = self.re.subprocess.run

        def spy(argv, *a, **k):
            calls.append(list(argv))
            return real_run(argv, *a, **k)

        saved = self.re.synthesize, self.re.subprocess.run
        self.re.synthesize = lambda prompt, entry, budget: (prompt + "    return None\n", "None", "stub", 0)
        self.re.subprocess.run = spy
        try:
            r = self.re.work((0, 1, False))
        finally:
            self.re.synthesize, self.re.subprocess.run = saved
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2:4], [os.path.abspath(self.re.RUNNER), "--script"])
        self.assertFalse(r["passed"])
        self.assertIn(r["reason"], ("assert", "exception"))


if __name__ == "__main__":
    unittest.main()
