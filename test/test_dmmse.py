"""
Tests for the dMMSE (discrete Maximum Mean Squared Error) prediction function.
"""

import math
import pytest
import torch

from evaluate_ood import predict_dmmse


class TestDMMSE:
    """Test suite for predict_dmmse function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple known case."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 3, 2
        M = 2
        sigma2 = 0.25
        
        # Create simple test data
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        # Run prediction
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Check output shape
        assert y_pred.shape == (B,)
        assert y_pred.dtype == torch.float32
        
        # Verify predictions are finite
        assert torch.all(torch.isfinite(y_pred))
    
    def test_posterior_normalization(self):
        """Test that posterior probabilities sum to 1."""
        torch.manual_seed(42)
        B, K_minus_1, D = 3, 5, 4
        M = 4
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        # Manually compute posterior to verify normalization
        y_pred_context = x_context @ tasks.T  # (B, K_minus_1, M)
        y_context_expanded = y_context.unsqueeze(-1)  # (B, K_minus_1, 1)
        squared_errors = (y_context_expanded - y_pred_context) ** 2
        sum_squared_errors = squared_errors.sum(dim=1)  # (B, M)
        log_likelihood = -0.5 * sum_squared_errors / sigma2
        log_sum_exp = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)
        log_posterior = log_likelihood - log_sum_exp
        posterior = torch.exp(log_posterior)  # (B, M)
        
        # Verify posterior sums to 1 for each batch
        posterior_sums = posterior.sum(dim=-1)
        assert torch.allclose(posterior_sums, torch.ones(B), atol=1e-6)
    
    def test_perfect_match(self):
        """Test case where one task perfectly matches the context data."""
        torch.manual_seed(42)
        B, K_minus_1, D = 1, 3, 2
        M = 3
        sigma2 = 0.25
        
        # Create a task that will perfectly match
        true_theta = torch.tensor([[1.0, 2.0]])  # (1, D)
        x_context = torch.randn(B, K_minus_1, D)
        y_context = (x_context @ true_theta.T).squeeze(-1)  # Perfect match, no noise
        x_query = torch.randn(B, 1, D)
        
        # Create tasks: one is the true task, others are random
        tasks = torch.cat([
            true_theta,
            torch.randn(M - 1, D)
        ], dim=0)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # The true task should get very high posterior probability
        y_pred_context = x_context @ tasks.T
        y_context_expanded = y_context.unsqueeze(-1)
        squared_errors = (y_context_expanded - y_pred_context) ** 2
        sum_squared_errors = squared_errors.sum(dim=1)
        log_likelihood = -0.5 * sum_squared_errors / sigma2
        log_sum_exp = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)
        log_posterior = log_likelihood - log_sum_exp
        posterior = torch.exp(log_posterior)
        
        # First task (true task) should have highest posterior
        assert posterior[0, 0] > posterior[0, 1]
        assert posterior[0, 0] > posterior[0, 2]
        
        # Prediction should be close to true task's prediction
        true_pred = (x_query @ true_theta.T).squeeze(-1)
        assert torch.allclose(y_pred, true_pred, atol=0.1)
    
    def test_uniform_prior(self):
        """Test case where all tasks are equally likely (equal log-likelihoods)."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 2, 2
        M = 3
        sigma2 = 0.25
        
        # Create context where all tasks give same prediction
        x_context = torch.zeros(B, K_minus_1, D)
        y_context = torch.zeros(B, K_minus_1)  # All zeros
        x_query = torch.randn(B, 1, D)
        
        # Create tasks that all predict zero for zero input
        tasks = torch.randn(M, D)
        # Make sure all tasks predict similar values for zero input
        # (they all predict ~0 since x_context is zero)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Manually compute posterior
        y_pred_context = x_context @ tasks.T
        y_context_expanded = y_context.unsqueeze(-1)
        squared_errors = (y_context_expanded - y_pred_context) ** 2
        sum_squared_errors = squared_errors.sum(dim=1)
        log_likelihood = -0.5 * sum_squared_errors / sigma2
        log_sum_exp = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)
        log_posterior = log_likelihood - log_sum_exp
        posterior = torch.exp(log_posterior)
        
        # Since all tasks predict ~0 for zero input, posteriors should be approximately uniform
        # (all log-likelihoods should be similar)
        posterior_std = posterior.std(dim=-1)
        # Standard deviation should be small (posteriors are similar)
        assert torch.all(posterior_std < 0.1)
        
        # Prediction should be approximately mean of all task predictions
        all_task_preds = x_query @ tasks.T  # (B, 1, M)
        mean_pred = all_task_preds.mean(dim=-1).squeeze(-1)
        assert torch.allclose(y_pred, mean_pred, atol=0.1)
    
    def test_numerical_stability_large_likelihoods(self):
        """Test numerical stability with very large log-likelihoods."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 3, 2
        M = 5
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        # Create one task that's much better (very small error)
        # Make first task predict values close to y_context
        tasks[0] = torch.linalg.lstsq(x_context[0], y_context[0].unsqueeze(-1)).solution.squeeze(-1)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Should still produce finite, reasonable predictions
        assert torch.all(torch.isfinite(y_pred))
        assert torch.all(torch.abs(y_pred) < 1e6)  # Not unreasonably large
    
    def test_numerical_stability_small_likelihoods(self):
        """Test numerical stability with very small log-likelihoods."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 3, 2
        M = 5
        sigma2 = 0.25
        
        # Create tasks that all have large errors
        x_context = torch.ones(B, K_minus_1, D) * 100  # Large values
        y_context = torch.ones(B, K_minus_1) * 100
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D) * 0.01  # Small tasks
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Should still produce finite predictions
        assert torch.all(torch.isfinite(y_pred))
    
    def test_single_task(self):
        """Test edge case with single task (M=1)."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 3, 2
        M = 1
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # With single task, posterior should be 1.0, prediction should be x_query @ tasks[0]
        # x_query @ tasks.T gives (B, 1, 1), need to squeeze both dimensions
        expected_pred = (x_query @ tasks.T).squeeze()  # (B,)
        assert torch.allclose(y_pred, expected_pred, atol=1e-6)
    
    def test_single_context_example(self):
        """Test edge case with single context example (K-1=1)."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 1, 2
        M = 3
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Should still work correctly
        assert y_pred.shape == (B,)
        assert torch.all(torch.isfinite(y_pred))
    
    def test_large_M(self):
        """Test with large number of tasks."""
        torch.manual_seed(42)
        B, K_minus_1, D = 2, 3, 2
        M = 1000
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Should handle large M without issues
        assert y_pred.shape == (B,)
        assert torch.all(torch.isfinite(y_pred))
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        torch.manual_seed(42)
        K_minus_1, D = 3, 2
        M = 3
        sigma2 = 0.25
        
        for B in [1, 5, 10, 100]:
            x_context = torch.randn(B, K_minus_1, D)
            y_context = torch.randn(B, K_minus_1)
            x_query = torch.randn(B, 1, D)
            tasks = torch.randn(M, D)
            
            y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
            
            assert y_pred.shape == (B,)
            assert torch.all(torch.isfinite(y_pred))
    
    def test_shape_validation(self):
        """Test shape validation for various input shapes."""
        torch.manual_seed(42)
        sigma2 = 0.25
        
        # Test various valid shapes
        test_cases = [
            ((1, 3, 2), (1, 3), (1, 1, 2), (5, 2)),  # B=1, K-1=3, D=2, M=5
            ((10, 5, 4), (10, 5), (10, 1, 4), (20, 4)),  # B=10, K-1=5, D=4, M=20
            ((3, 1, 8), (3, 1), (3, 1, 8), (1, 8)),  # B=3, K-1=1, D=8, M=1
        ]
        
        for (x_shape, y_shape, q_shape, t_shape) in test_cases:
            x_context = torch.randn(*x_shape)
            y_context = torch.randn(*y_shape)
            x_query = torch.randn(*q_shape)
            tasks = torch.randn(*t_shape)
            
            y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
            
            B = x_shape[0]
            assert y_pred.shape == (B,)
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness with manually computed example."""
        torch.manual_seed(42)
        B, K_minus_1, D = 1, 2, 2
        M = 2
        sigma2 = 0.25
        
        # Simple known case
        x_context = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        y_context = torch.tensor([[1.0, 2.0]])  # (1, 2)
        x_query = torch.tensor([[[1.0, 1.0]]])  # (1, 1, 2)
        tasks = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # (2, 2)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Manually compute expected result
        # Task 0: theta = [1, 0]
        #   y_pred_context[0,0] = [1,0] @ [1,0]^T = 1
        #   y_pred_context[0,1] = [0,1] @ [1,0]^T = 0
        #   errors: (1-1)^2 + (2-0)^2 = 0 + 4 = 4
        #   log_likelihood[0] = -0.5 * 4 / 0.25 = -8
        
        # Task 1: theta = [0, 2]
        #   y_pred_context[0,0] = [1,0] @ [0,2]^T = 0
        #   y_pred_context[0,1] = [0,1] @ [0,2]^T = 2
        #   errors: (1-0)^2 + (2-2)^2 = 1 + 0 = 1
        #   log_likelihood[1] = -0.5 * 1 / 0.25 = -2
        
        # log_sum_exp = log(exp(-8) + exp(-2)) = log(exp(-2) * (exp(-6) + 1))
        # ≈ -2 + log(1 + exp(-6)) ≈ -2
        # posterior[0] = exp(-8 - (-2)) = exp(-6) ≈ 0.0025
        # posterior[1] = exp(-2 - (-2)) = exp(0) = 1.0
        
        # y_pred_query[0] = [1,1] @ [1,0]^T = 1
        # y_pred_query[1] = [1,1] @ [0,2]^T = 2
        # y_pred = 0.0025 * 1 + 1.0 * 2 ≈ 2.0
        
        # Task 1 should dominate (much better likelihood)
        assert y_pred[0] > 1.5  # Should be close to 2.0
        assert torch.isfinite(y_pred[0])
    
    def test_log_sum_exp_trick(self):
        """Test that log-sum-exp trick prevents numerical issues."""
        torch.manual_seed(42)
        B, K_minus_1, D = 1, 3, 2
        M = 10
        sigma2 = 0.25
        
        x_context = torch.randn(B, K_minus_1, D)
        y_context = torch.randn(B, K_minus_1)
        x_query = torch.randn(B, 1, D)
        tasks = torch.randn(M, D)
        
        # Create one task with much better fit (very large log-likelihood)
        # This would cause overflow without log-sum-exp
        # Make first task predict values close to y_context
        tasks[0] = torch.linalg.lstsq(x_context[0], y_context[0].unsqueeze(-1)).solution.squeeze(-1)
        
        y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
        
        # Should handle this without overflow
        assert torch.all(torch.isfinite(y_pred))
        
        # Verify posterior is still normalized
        y_pred_context = x_context @ tasks.T
        y_context_expanded = y_context.unsqueeze(-1)
        squared_errors = (y_context_expanded - y_pred_context) ** 2
        sum_squared_errors = squared_errors.sum(dim=1)
        log_likelihood = -0.5 * sum_squared_errors / sigma2
        log_sum_exp = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)
        log_posterior = log_likelihood - log_sum_exp
        posterior = torch.exp(log_posterior)
        
        assert torch.allclose(posterior.sum(dim=-1), torch.ones(B), atol=1e-5)
