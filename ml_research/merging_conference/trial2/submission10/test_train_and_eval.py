import unittest
import torch
import torch.nn as nn
from train_and_eval import (
    MultiTaskModel, 
    compute_clean_fisher_estimates, 
    run_test_time_adaptation, 
    get_merged_state_dict,
    get_expert_state_dict
)

class TestTrainAndEval(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        
        # Create a tiny model for fast testing
        self.num_tasks = 3
        self.num_classes = 10
        self.model = MultiTaskModel(num_tasks=self.num_tasks, num_classes=self.num_classes).to(self.device)
        
        # Mock pre-trained encoder state
        self.encoder_pre_state = {k: v.clone() for k, v in self.model.encoder.state_dict().items()}
        self.encoder_pre_params = set(name for name, _ in self.model.encoder.named_parameters())
        
        # Mock expert states (slightly perturbed from pre-trained)
        self.encoder_experts_states = []
        self.expert_heads_weights = []
        self.task_vectors = []
        
        for k in range(self.num_tasks):
            # Expert state
            expert_state = {}
            for name, val in self.encoder_pre_state.items():
                is_param = name in self.encoder_pre_params
                if val.is_floating_point() and is_param:
                    expert_state[name] = val + torch.randn_like(val) * 0.01
                else:
                    expert_state[name] = val.clone()
            self.encoder_experts_states.append(expert_state)
            
            # Head weights
            self.expert_heads_weights.append({
                "weight": torch.randn_like(self.model.heads[k].weight),
                "bias": torch.randn_like(self.model.heads[k].bias)
            })
            
            # Task vector
            vec = {}
            for name in self.encoder_pre_state.keys():
                if self.encoder_pre_state[name].is_floating_point() and "num_batches_tracked" not in name:
                    vec[name] = expert_state[name] - self.encoder_pre_state[name]
                else:
                    vec[name] = torch.zeros_like(self.encoder_pre_state[name])
            self.task_vectors.append(vec)

    def test_multitask_model_forward(self):
        # Generate a random batch of 3-channel images
        x = torch.randn(4, 3, 28, 28)
        # Check forward pass for each task head
        for k in range(self.num_tasks):
            logits = self.model(x, k)
            self.assertEqual(logits.shape, (4, self.num_classes))

    def test_clean_fisher_estimates(self):
        # Create a mock train DataLoader
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 16
            def __getitem__(self, idx):
                # Return random 3-channel image and random label
                return torch.randn(3, 28, 28), torch.randint(0, 10, (1,)).item()
                
        mock_loaders = [torch.utils.data.DataLoader(MockDataset(), batch_size=4) for _ in range(self.num_tasks)]
        
        fisher = compute_clean_fisher_estimates(
            model=self.model,
            encoder_pre_state=self.encoder_pre_state,
            encoder_pre_params=self.encoder_pre_params,
            encoder_experts_states=self.encoder_experts_states,
            expert_heads_weights=self.expert_heads_weights,
            task_vectors=self.task_vectors,
            train_loaders=mock_loaders,
            device=self.device,
            num_tasks=self.num_tasks,
            num_samples=8
        )
        
        # Verify keys in Fisher estimates
        for k in range(self.num_tasks):
            self.assertIn(f"heads.{k}.weight", fisher)
            self.assertIn(f"heads.{k}.bias", fisher)
            self.assertEqual(fisher[f"heads.{k}.weight"].shape, self.expert_heads_weights[k]["weight"].shape)
            self.assertEqual(fisher[f"heads.{k}.bias"].shape, self.expert_heads_weights[k]["bias"].shape)
            
        self.assertIn("lambdas", fisher)
        self.assertEqual(fisher["lambdas"].shape, (self.num_tasks,))

    def test_test_time_adaptation_methods(self):
        # Create mock TTA loaders
        class MockTTADataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return torch.randn(3, 28, 28), torch.randint(0, 10, (1,)).item()
                
        mock_tta_loaders = [torch.utils.data.DataLoader(MockTTADataset(), batch_size=4) for _ in range(self.num_tasks)]
        
        # Set up a small clean Fisher estimate
        clean_fisher = {}
        for k in range(self.num_tasks):
            clean_fisher[f"heads.{k}.weight"] = torch.ones_like(self.expert_heads_weights[k]["weight"]) * 0.1
            clean_fisher[f"heads.{k}.bias"] = torch.ones_like(self.expert_heads_weights[k]["bias"]) * 0.1
        clean_fisher["lambdas"] = torch.ones(self.num_tasks) * 0.5
        
        # Test BF-ASAM (f-asam)
        lambdas, adapted_heads = run_test_time_adaptation(
            model=self.model,
            encoder_pre_state=self.encoder_pre_state,
            encoder_pre_params=self.encoder_pre_params,
            encoder_experts_states=self.encoder_experts_states,
            expert_heads_weights=self.expert_heads_weights,
            task_vectors=self.task_vectors,
            tta_loaders=mock_tta_loaders,
            method="f-asam",
            device=self.device,
            clean_fisher_estimates=clean_fisher,
            num_tasks=self.num_tasks,
            steps=2,
            rho=0.05
        )
        
        self.assertEqual(lambdas.shape, (self.num_tasks,))
        self.assertEqual(len(adapted_heads), self.num_tasks * 2) # weight and bias for each of 3 tasks
        
        # Test R-BF-SAM (r-f-sam)
        lambdas_r, adapted_heads_r = run_test_time_adaptation(
            model=self.model,
            encoder_pre_state=self.encoder_pre_state,
            encoder_pre_params=self.encoder_pre_params,
            encoder_experts_states=self.encoder_experts_states,
            expert_heads_weights=self.expert_heads_weights,
            task_vectors=self.task_vectors,
            tta_loaders=mock_tta_loaders,
            method="r-f-sam",
            device=self.device,
            clean_fisher_estimates=clean_fisher,
            num_tasks=self.num_tasks,
            steps=2,
            rho=0.05,
            conf_threshold=0.3
        )
        
        self.assertEqual(lambdas_r.shape, (self.num_tasks,))

if __name__ == "__main__":
    unittest.main()
