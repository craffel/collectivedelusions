#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
# ]
# ///

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from run_optimization import extract_answer, parse_max_past_iterates, EQUATIONS, approximate_mse, fallback_extract_answer, run_experiment

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


class TestResume(unittest.TestCase):
    def test_resume_reconstruction(self):
        # Create a mock initial_results dict with 2 completed rounds
        mock_initial = {
            "rounds": [
                {
                    "round": 1,
                    "guesses": [{"m": 5.62, "b": 0.65}],
                    "chosen": {"m": 5.62, "b": 0.65, "approx_mse": 0.15} # < 0.18
                },
                {
                    "round": 2,
                    "guesses": [{"m": 5.62, "b": 0.65}],
                    "chosen": {"m": 5.62, "b": 0.65, "approx_mse": 0.12} # < 0.18
                }
            ]
        }
        
        # Mock Client and models
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_response.text = "$\\boxed{(5.62, 0.65)}$"
        
        with patch('google.genai.Client', return_value=mock_client):
            # Run experiment for 3 steps, resuming from 2 completed rounds.
            # Round 3 should execute, and since its MSE will be < 0.18, 
            # the consecutive_low_mse count will hit 3, triggering early stopping!
            results = run_experiment(
                generator_model="mock-gen",
                judge_model="mock-judge",
                n_steps=3,
                n_guesses=1,
                initial_results=mock_initial,
                early_stopping_mse=0.18
            )
            
            # Since Round 1 and Round 2 were already completed, and Round 3 completed
            # and triggered early stopping immediately, rounds should have exactly 3 items.
            self.assertEqual(len(results["rounds"]), 3)
            self.assertEqual(results["rounds"][2]["round"], 3)
            self.assertEqual(len(results["final_iterates"]), 3)


if __name__ == "__main__":
    unittest.main()
