#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
# ]
# ///

import unittest
from unittest.mock import MagicMock
import numpy as np
from run_optimization import extract_answer, parse_max_past_iterates, EQUATIONS, approximate_mse, fallback_extract_answer

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


class TestParseMaxPastIterates(unittest.TestCase):
    def test_valid_all(self):
        self.assertEqual(parse_max_past_iterates("all"), "all")
        self.assertEqual(parse_max_past_iterates("ALL"), "all")
        self.assertEqual(parse_max_past_iterates("All"), "all")

    def test_valid_integers(self):
        self.assertEqual(parse_max_past_iterates("0"), 0)
        self.assertEqual(parse_max_past_iterates("3"), 3)
        self.assertEqual(parse_max_past_iterates("100"), 100)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            parse_max_past_iterates("-5")
        with self.assertRaises(ValueError):
            parse_max_past_iterates("invalid")
        with self.assertRaises(ValueError):
            parse_max_past_iterates("3.5")


class TestEquationsAndMSE(unittest.TestCase):
    def test_equations_are_strings(self):
        # Make sure EQUATIONS dictionary holds valid string prompt segments
        for opt_id, text in EQUATIONS.items():
            self.assertTrue(isinstance(text, str))
            self.assertTrue(len(text) > 0)

    def test_approximate_mse_runs(self):
        # Test approximate_mse evaluates cleanly and returns valid float MSE values
        mse = approximate_mse(1.0, 2.0)
        self.assertTrue(isinstance(mse, float))
        self.assertTrue(mse >= 0.0)


class TestFallbackExtraction(unittest.TestCase):
    def test_fallback_extract_answer_logic(self):
        # Mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        # Test case 1: fallback returns "NONE" (preventing hallucination)
        mock_response.text = "NONE"
        res = fallback_extract_answer(mock_client, "some input text")
        self.assertIsNone(res)
        
        # Test case 2: fallback returns "none" in lowercase/whitespace
        mock_response.text = "  none  \n"
        res = fallback_extract_answer(mock_client, "some input text")
        self.assertIsNone(res)
        
        # Test case 3: fallback returns valid boxed tuple
        mock_response.text = "$\\boxed{(5.62, 0.65)}$"
        res = fallback_extract_answer(mock_client, "some input text")
        self.assertEqual(res, (5.62, 0.65))


if __name__ == "__main__":
    unittest.main()
