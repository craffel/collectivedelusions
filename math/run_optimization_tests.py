#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
# ]
# ///

import unittest
from run_optimization import extract_answer

class TestExtractAnswer(unittest.TestCase):
    def test_standard_case(self):
        self.assertEqual(extract_answer(r"$\boxed{(1.5, 2.0)}$"), (1.5, 2.0))
        self.assertEqual(extract_answer(r"\boxed{(1.5, 2.0)}"), (1.5, 2.0))

    def test_negative_numbers(self):
        self.assertEqual(extract_answer(r"$\boxed{(-1.5, -2.5)}$"), (-1.5, -2.5))
        self.assertEqual(extract_answer(r"$\boxed{(-1.5, 2.5)}$"), (-1.5, 2.5))
        self.assertEqual(extract_answer(r"$\boxed{(1.5, -2.5)}$"), (1.5, -2.5))

    def test_scientific_notation(self):
        self.assertEqual(extract_answer(r"$\boxed{(1.2e-3, -4.5e6)}$"), (1.2e-3, -4.5e6))
        self.assertEqual(extract_answer(r"$\boxed{(1.2E3, 4.5e+2)}$"), (1.2e3, 450.0))

    def test_whitespace_tolerance(self):
        self.assertEqual(extract_answer(r"\boxed{   (  1.5  ,   -2.0  )   }"), (1.5, -2.0))
        self.assertEqual(extract_answer(r"  \boxed  {  (  1.5  ,  2.0  )  }  "), (1.5, 2.0))

    def test_multiple_matches_returns_last(self):
        text = r"First attempt: $\boxed{(1.0, 2.0)}$ and then we refine to $\boxed{(3.5, 4.5)}$"
        self.assertEqual(extract_answer(text), (3.5, 4.5))

    def test_leading_decimal(self):
        self.assertEqual(extract_answer(r"\boxed{(-.5, .75)}"), (-0.5, 0.75))

    def test_no_match_invalid_formats(self):
        self.assertIsNone(extract_answer(r"$\boxed{(m, b)}$"))
        self.assertIsNone(extract_answer(r"\boxed{(1.0, 2.0)"))
        self.assertIsNone(extract_answer(r"\boxed{1.0, 2.0}"))
        self.assertIsNone(extract_answer("Just some random text"))
        self.assertIsNone(extract_answer(r"\boxed{(1.0)}"))
        self.assertIsNone(extract_answer(r"\boxed{(1. 0, 2.0)}"))

if __name__ == "__main__":
    unittest.main()
