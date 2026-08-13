#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
#   "openrouter",
#   "httpx",
# ]
# ///

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import openrouter
import openrouter.errors
import httpx
from run_optimization import (
    extract_answer,
    parse_max_past_iterates,
    EQUATIONS,
    approximate_mse,
    fallback_extract,
    run_experiment,
    run_first_step_performance,
    instantiate_clients,
    TUPLE_EXTRACTION_PROMPT,
    GoogleClient,
    OpenRouterClient
)

class TestExtractAnswer(unittest.TestCase):
    def test_standard_case(self):
        self.assertIsNone(extract_answer(None))
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
    def test_fallback_extract_logic(self):
        # Mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        fallback_client = GoogleClient("gemini-3.5-flash", client=mock_client)
        
        # Test case 1: fallback returns "NONE" (preventing hallucination)
        mock_response.text = "NONE"
        res = fallback_extract("some input text", TUPLE_EXTRACTION_PROMPT, extract_answer, fallback_client)
        self.assertIsNone(res)
        
        # Test case 2: fallback returns "none" in lowercase/whitespace
        mock_response.text = "  none  \n"
        res = fallback_extract("some input text", TUPLE_EXTRACTION_PROMPT, extract_answer, fallback_client)
        self.assertIsNone(res)
        
        # Test case 3: fallback returns valid boxed tuple
        mock_response.text = "$\\boxed{(5.62, 0.65)}$"
        res = fallback_extract("some input text", TUPLE_EXTRACTION_PROMPT, extract_answer, fallback_client)
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
                generator_model="google/mock-gen",
                judge_model="google/mock-judge",
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


class TestMSEPromptOptions(unittest.TestCase):
    def test_generator_and_judge_prompt_formatting_with_mse(self):
        # Verify generator and judge prompts can dynamically accept and show MSE
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_response.text = "$\\boxed{(5.62, 0.65)}$"
        
        mock_initial = {
            "rounds": [
                {
                    "round": 1,
                    "guesses": [{"m": 5.62, "b": 0.65}],
                    "chosen": {"m": 5.62, "b": 0.65, "approx_mse": 0.15}
                }
            ]
        }
        
        with patch('google.genai.Client', return_value=mock_client):
            # Run experiment for 2 steps, resuming from 1 completed round
            results = run_experiment(
                generator_model="google/mock-gen",
                judge_model="google/mock-judge",
                n_steps=2,
                n_guesses=1,
                initial_results=mock_initial,
                generator_show_mse=True,
                judge_show_mse=True
            )
            
            # Inspect mock calls to verify prompts contains the MSE string
            calls = mock_client.models.generate_content.call_args_list
            
            # Find prompt text from calls list
            prompts = [call[1]['contents'] for call in calls]
            
            # Generator prompt should contain: "with MSE = 0.150000" (reconstructed from initial round chosen)
            generator_prompts = [p for p in prompts if "Below are the values found in past iterations." in p]
            self.assertTrue(len(generator_prompts) >= 1)
            self.assertTrue("with MSE =" in generator_prompts[0])
            
            # Judge prompt should contain: "with MSE = <value>" for proposed guess
            judge_prompts = [p for p in prompts if "Below are 1 possible values" in p]
            self.assertTrue(len(judge_prompts) >= 1)
            self.assertTrue("with MSE =" in judge_prompts[0])


class TestMSEApproximations(unittest.TestCase):
    def test_generator_and_judge_mse_approximations(self):
        # Verify generator and judge MSE approximations are triggered and correctly recorded in results
        mock_client = MagicMock()
        
        mock_initial = {
            "rounds": [
                {
                    "round": 1,
                    "guesses": [{"m": 5.62, "b": 0.65}],
                    "chosen": {
                        "m": 5.62,
                        "b": 0.65,
                        "approx_mse": 0.15,
                        "approximated_mse_by_generator": 0.150000
                    }
                }
            ]
        }
        
        with patch('google.genai.Client', return_value=mock_client):
            mock_resp_approx = MagicMock(text="$\\boxed{0.125}$")
            mock_resp_tuple = MagicMock(text="$\\boxed{(5.62, 0.65)}$")
            
            # Sequenced side effect for mock calls matching proactive guess approximation flow:
            mock_client.models.generate_content.side_effect = [
                mock_resp_tuple,  # Generator guess generation
                mock_resp_approx, # Generator approximation of the newly generated guess
                mock_resp_approx, # Judge approximation of the proposed guess
                mock_resp_tuple   # Judge selection
            ]
            
            results = run_experiment(
                generator_model="google/mock-gen",
                judge_model="google/mock-judge",
                n_steps=2,
                n_guesses=1,
                initial_results=mock_initial,
                generator_approximate_mse=True,
                judge_approximate_mse=True
            )
            
            # Assertions to verify that approximated values are saved inside results!
            round_2 = results["rounds"][1]
            
            # Judge approximation for the single guess in Round 2 should be parsed and saved
            self.assertEqual(round_2["guesses"][0]["approximated_mse_by_judge"], 0.125)
            
            # Generator approximation for the past iterate in Round 2 should be parsed and saved
            self.assertEqual(round_2["guesses"][0]["approximated_mse_by_generator"], 0.125)
            
            # Verify that the subsequent prompts contains the LLM-approximated MSE values!
            calls = mock_client.models.generate_content.call_args_list
            prompts = [call[1]['contents'] for call in calls]
            
            # Find generator and judge prompts
            generator_prompts = [p for p in prompts if "Below are the values found in past iterations." in p]
            self.assertTrue(len(generator_prompts) >= 1)
            # Generator prompt should contain the loaded previous round's generator approximation: "with MSE = 0.150000"
            self.assertTrue("with MSE = 0.150000" in generator_prompts[0])
            
            judge_prompts = [p for p in prompts if "Below are 1 possible values" in p]
            self.assertTrue(len(judge_prompts) >= 1)
            # Judge prompt should contain the current round's judge approximation: "with MSE = 0.125000"
            self.assertTrue("with MSE = 0.125000" in judge_prompts[0])


class TestOpenRouterIntegration(unittest.TestCase):
    def test_provider_parsing(self):
        from run_optimization import parse_provider_and_model
        
        provider, name = parse_provider_and_model("openrouter/nvidia/nemotron")
        self.assertEqual(provider, "openrouter")
        self.assertEqual(name, "nvidia/nemotron")
        
        provider, name = parse_provider_and_model("google/gemini-2.5-flash")
        self.assertEqual(provider, "google")
        self.assertEqual(name, "gemini-2.5-flash")
        
        provider, name = parse_provider_and_model("mse")
        self.assertEqual(provider, "")
        self.assertEqual(name, "mse")

        # Ensure that no API prefix throws a ValueError
        with self.assertRaises(ValueError):
            parse_provider_and_model("gemini-3.5-flash")
            
        # Ensure that an invalid API prefix throws a ValueError
        with self.assertRaises(ValueError):
            parse_provider_and_model("aws/claude")

class TestInstantiateClients(unittest.TestCase):
    @patch('run_optimization.genai.Client')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('run_optimization.openrouter.OpenRouter')
    def test_instantiate_clients(self, mock_openrouter, mock_genai):
        gen_client, judge_client, fallback_client = instantiate_clients(
            "google/gemini-2.5-flash",
            "openrouter/nvidia/nemotron"
        )
        self.assertIsInstance(gen_client, GoogleClient)
        self.assertIsInstance(judge_client, OpenRouterClient)
        self.assertIsInstance(fallback_client, GoogleClient)
        self.assertEqual(fallback_client.model_name, "gemini-3.5-flash")

    @patch('run_optimization.genai.Client')
    def test_instantiate_clients_mse_judge(self, mock_genai):
        gen_client, judge_client, fallback_client = instantiate_clients(
            "google/gemini-2.5-flash",
            "mse"
        )
        self.assertIsInstance(gen_client, GoogleClient)
        self.assertIsNone(judge_client)
        self.assertIsInstance(fallback_client, GoogleClient)


class TestGoogleClient(unittest.TestCase):
    def setUp(self):
        # Patch random.randint to return 0 to make retry delays deterministic (1s)
        self.randint_patcher = patch('run_optimization.random.randint', return_value=0)
        self.randint_patcher.start()
        self.addCleanup(self.randint_patcher.stop)

    @patch('run_optimization.genai.Client')
    def test_init(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        
        self.assertEqual(client.model_name, "gemini-3.5-flash")
        mock_genai_client_class.assert_called_once()
        self.assertEqual(client.client, mock_client)

    @patch('run_optimization.genai.Client')
    def test_generate_success(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello world")
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-3.5-flash",
            contents="test prompt"
        )

    @patch('time.sleep')
    @patch('run_optimization.genai.Client')
    def test_generate_server_error_retry(self, mock_genai_client_class, mock_sleep):
        import run_optimization
        errors = run_optimization.genai.errors
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello after retry"
        
        # Raise ServerError first, then return successful response
        mock_client.models.generate_content.side_effect = [
            errors.ServerError(500, {}),
            mock_response
        ]
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after retry")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    @patch('run_optimization.genai.Client')
    def test_generate_rate_limit_retry(self, mock_genai_client_class, mock_sleep):
        import run_optimization
        errors = run_optimization.genai.errors
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello after rate limit"
        
        # Create an APIError with e.code == 429
        api_error = errors.APIError(429, {})
        
        mock_client.models.generate_content.side_effect = [
            api_error,
            mock_response
        ]
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after rate limit")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('run_optimization.genai.Client')
    def test_generate_other_api_error_raises(self, mock_genai_client_class):
        import run_optimization
        errors = run_optimization.genai.errors
        mock_client = MagicMock()
        
        # Create an APIError with e.code == 400
        api_error = errors.APIError(400, {})
        
        mock_client.models.generate_content.side_effect = api_error
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        with self.assertRaises(errors.APIError):
            client.generate("test prompt")

    @patch('time.sleep')
    @patch('run_optimization.genai.Client')
    def test_generate_httpx_http_error_retry(self, mock_genai_client_class, mock_sleep):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello after httpx error"
        
        # Raise httpx.HTTPError, then return successful response
        mock_client.models.generate_content.side_effect = [
            httpx.HTTPError("Network failure"),
            mock_response
        ]
        mock_genai_client_class.return_value = mock_client
        
        client = GoogleClient("gemini-3.5-flash")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after httpx error")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('run_optimization.genai.Client')
    def test_generate_jitter_behavior(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Success"
        
        # Temporarily stop the setUp patcher so we can test actual randint patching
        self.randint_patcher.stop()
        
        try:
            with patch('run_optimization.random.randint') as mock_randint, \
                 patch('time.sleep') as mock_sleep:
                mock_randint.return_value = 1
                mock_client.models.generate_content.side_effect = [
                    httpx.HTTPError("Error"),
                    mock_response
                ]
                mock_genai_client_class.return_value = mock_client
                client = GoogleClient("gemini-3.5-flash")
                client.generate("test prompt")
                
                # Assert random.randint was called with (-1, 1)
                mock_randint.assert_called_once_with(-1, 1)
                # Delay is 1 + jitter (1) = 2
                mock_sleep.assert_called_once_with(2)
        finally:
            # Re-start setUp patcher to clean up
            self.randint_patcher.start()


class TestOpenRouterClient(unittest.TestCase):
    def setUp(self):
        # Patch random.randint to return 0 to make retry delays deterministic (1s)
        self.randint_patcher = patch('run_optimization.random.randint', return_value=0)
        self.randint_patcher.start()
        self.addCleanup(self.randint_patcher.stop)

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_init(self, mock_openrouter_client_class):
        mock_client = MagicMock()
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        
        self.assertEqual(client.model_name, "nvidia/nemotron")
        self.assertEqual(client.api_key, "fake_key")
        self.assertEqual(client.client, mock_client)
        mock_openrouter_client_class.assert_called_once_with(api_key="fake_key")

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_success(self, mock_openrouter_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello from OpenRouter"))
        ]
        mock_client.chat.send.return_value = mock_response
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello from OpenRouter")
        mock_client.chat.send.assert_called_once_with(
            model="nvidia/nemotron",
            messages=[{"role": "user", "content": "test prompt"}],
            provider={
                "sort": "price"
            }
        )

    @patch('time.sleep')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_rate_limit_retry(self, mock_openrouter_client_class, mock_sleep):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello after rate limit"))
        ]
        
        # Raise TooManyRequestsResponseError, then return successful response
        mock_client.chat.send.side_effect = [
            openrouter.errors.TooManyRequestsResponseError(MagicMock(), MagicMock()),
            mock_response
        ]
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after rate limit")
        self.assertEqual(mock_client.chat.send.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_server_error_retry(self, mock_openrouter_client_class, mock_sleep):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello after server error"))
        ]
        
        # Raise InternalServerResponseError, then return successful response
        mock_client.chat.send.side_effect = [
            openrouter.errors.InternalServerResponseError(MagicMock(), MagicMock()),
            mock_response
        ]
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after server error")
        self.assertEqual(mock_client.chat.send.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_response_validation_error_retry(self, mock_openrouter_client_class, mock_sleep):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello after validation error"))
        ]
        
        # Raise ResponseValidationError, then return successful response
        mock_client.chat.send.side_effect = [
            openrouter.errors.ResponseValidationError("Response validation failed", MagicMock(), Exception()),
            mock_response
        ]
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after validation error")
        self.assertEqual(mock_client.chat.send.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_httpx_http_error_retry(self, mock_openrouter_client_class, mock_sleep):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello after httpx error"))
        ]
        
        # Raise httpx.HTTPError, then return successful response
        mock_client.chat.send.side_effect = [
            httpx.HTTPError("Network failure"),
            mock_response
        ]
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        res = client.generate("test prompt")
        
        self.assertEqual(res, "Hello after httpx error")
        self.assertEqual(mock_client.chat.send.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake_key'})
    @patch('openrouter.OpenRouter')
    def test_generate_api_error_other_raises(self, mock_openrouter_client_class):
        mock_client = MagicMock()
        
        # Raise a general OpenRouterError (with a mock raw_response, etc.)
        mock_client.chat.send.side_effect = openrouter.errors.OpenRouterError("Some bad error", MagicMock())
        mock_openrouter_client_class.return_value = mock_client
        
        client = OpenRouterClient("nvidia/nemotron")
        with self.assertRaises(openrouter.errors.OpenRouterError) as ctx:
            client.generate("test prompt")
        self.assertEqual(ctx.exception.message, "Some bad error")
class TestFirstStepPerformance(unittest.TestCase):
    @patch('run_optimization.genai.Client')
    @patch('run_optimization.GoogleClient')
    def test_first_step_performance_basic(self, mock_google_client_class, mock_genai_client_class):
        mock_genai_client_class.return_value = MagicMock()
        mock_fallback_client = MagicMock()
        mock_fallback_client.generate.return_value = r"\boxed{(6.0, 0.2)}"
        mock_google_client_class.return_value = mock_fallback_client

        def mock_generate_and_parse(client, prompt, primary_parser, **kwargs):
            if "generator first-step run" in kwargs.get("label", ""):
                return (6.0, 0.2)
            if "judge first-step selection" in kwargs.get("label", ""):
                return (6.046, 0.1409)
            return (6.0, 0.2)

        # Test with LLM models for generator and judge
        with patch('run_optimization.generate_and_parse_with_fallback', side_effect=mock_generate_and_parse):
            results = run_first_step_performance(
                generator_model="google/gemini-2.5-flash",
                judge_model="google/gemini-2.5-flash",
                n_steps=3,
                n_guesses=5,
                equation_option=1
            )
        self.assertIn("generator_results", results)
        self.assertIn("judge_results", results)
        self.assertEqual(len(results["generator_results"]), 3)
        self.assertEqual(len(results["judge_results"]), 3)

        for gen_res in results["generator_results"]:
            self.assertEqual(gen_res["m"], 6.0)
            self.assertEqual(gen_res["b"], 0.2)
            self.assertIn("mse", gen_res)
            self.assertGreater(gen_res["mse"], 0.0)

        for judge_res in results["judge_results"]:
            guesses = judge_res["guesses"]
            self.assertEqual(len(guesses), 5)
            self.assertEqual((guesses[0]["m"], guesses[0]["b"]), (6.046, 0.1409))
            self.assertEqual((guesses[1]["m"], guesses[1]["b"]), (6.103, 1.0))
            for g in guesses[2:]:
                self.assertGreaterEqual(g["m"], -10.0)
                self.assertLessEqual(g["m"], 10.0)
                self.assertGreaterEqual(g["b"], -10.0)
                self.assertLessEqual(g["b"], 10.0)
                # Verify formatting to 3 decimal places
                self.assertEqual(g["m"], round(g["m"], 3))
                self.assertEqual(g["b"], round(g["b"], 3))
            self.assertTrue(judge_res["picked_target"])

        self.assertEqual(results["summary"]["judge_target_picks"], 3)
        self.assertEqual(results["summary"]["judge_target_pick_rate"], 1.0)

    @patch('run_optimization.genai.Client')
    @patch('run_optimization.GoogleClient')
    def test_first_step_performance_llm_judge(self, mock_google_client_class, mock_genai_client_class):
        mock_genai_client_class.return_value = MagicMock()
        mock_fallback_client = MagicMock()
        mock_google_client_class.return_value = mock_fallback_client

        # Mock generate_and_parse_with_fallback for generator and judge
        def mock_generate_and_parse(client, prompt, primary_parser, **kwargs):
            if "first-step run" in kwargs.get("label", ""):
                return (6.0, 0.2)
            if "judge first-step selection" in kwargs.get("label", ""):
                return (6.046, 0.1409)
            return (6.0, 0.2)

        with patch('run_optimization.generate_and_parse_with_fallback', side_effect=mock_generate_and_parse):
            results = run_first_step_performance(
                generator_model="google/gemini-2.5-flash",
                judge_model="google/gemini-2.5-flash",
                n_steps=2,
                n_guesses=4,
                equation_option=1
            )
        self.assertEqual(len(results["judge_results"]), 2)
        for judge_res in results["judge_results"]:
            self.assertEqual(len(judge_res["guesses"]), 4)
            self.assertTrue(judge_res["picked_target"])


if __name__ == "__main__":
    unittest.main()
