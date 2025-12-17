"""
Tests for the predict.py module, specifically the create_prompt function.
"""

import math
import pytest
import torch

from predict import create_prompt


class TestCreatePrompt:
    """Test suite for create_prompt function."""
    
    def test_basic_functionality(self):
        """Test basic prompt creation with single theta."""
        D = 8
        theta = torch.randn(D)
        n_examples = 5
        n_query = 3
        
        x_context, y_context, x_query = create_prompt(
            [(theta, n_examples)],
            n_query=n_query,
            D=D
        )
        
        # Check shapes
        assert x_context.shape == (n_examples, D)
        assert y_context.shape == (n_examples,)
        assert x_query.shape == (n_query, D)
        
        # Check that y values are computed correctly (with noise)
        y_expected = x_context @ theta
        # Allow for noise variance
        noise_std = math.sqrt(0.125)  # default sigma2
        assert torch.allclose(y_context, y_expected, atol=5 * noise_std)
    
    def test_multiple_thetas(self):
        """Test prompt creation with multiple thetas."""
        D = 8
        theta1 = torch.randn(D)
        theta2 = torch.randn(D)
        theta3 = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta1, 2), (theta2, 3), (theta3, 1)],
            n_query=2,
            D=D
        )
        
        # Should have 2 + 3 + 1 = 6 context examples
        assert x_context.shape == (6, D)
        assert y_context.shape == (6,)
        assert x_query.shape == (2, D)
        
        # Check that examples from different thetas are different
        # First 2 examples should match theta1
        y1_expected = x_context[:2] @ theta1
        assert torch.allclose(y_context[:2], y1_expected, atol=5 * math.sqrt(0.125))
        
        # Next 3 examples should match theta2
        y2_expected = x_context[2:5] @ theta2
        assert torch.allclose(y_context[2:5], y2_expected, atol=5 * math.sqrt(0.125))
        
        # Last example should match theta3
        y3_expected = x_context[5:6] @ theta3
        assert torch.allclose(y_context[5:6], y3_expected, atol=5 * math.sqrt(0.125))
    
    def test_query_theta_default(self):
        """Test that query_theta defaults to last theta."""
        D = 8
        theta1 = torch.randn(D)
        theta2 = torch.randn(D)
        
        # Without specifying query_theta, should use last theta (theta2)
        x_context, y_context, x_query = create_prompt(
            [(theta1, 2), (theta2, 2)],
            n_query=3,
            D=D,
            seed=42
        )
        
        # Query points should be generated from theta2
        y_query_expected = x_query @ theta2
        # Note: We can't directly check y_query since it's not returned,
        # but we can verify the function runs correctly
    
    def test_query_theta_explicit(self):
        """Test explicit query_theta specification."""
        D = 8
        theta1 = torch.randn(D)
        theta2 = torch.randn(D)
        query_theta = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta1, 2), (theta2, 2)],
            query_theta=query_theta,
            n_query=3,
            D=D
        )
        
        assert x_query.shape == (3, D)
        # Query points are generated but y values aren't returned,
        # so we just verify shapes are correct
    
    def test_noise_variance(self):
        """Test that noise variance is applied correctly."""
        D = 8
        theta = torch.randn(D)
        sigma2 = 0.25  # Larger noise for easier detection
        n_examples = 100  # More examples for better statistics
        
        x_context, y_context, x_query = create_prompt(
            [(theta, n_examples)],
            n_query=1,
            D=D,
            sigma2=sigma2,
            seed=42
        )
        
        # Check that noise is present
        y_noiseless = x_context @ theta
        noise = y_context - y_noiseless
        noise_var = torch.var(noise).item()
        
        # Should be close to sigma2 (allow some variance due to finite samples)
        assert abs(noise_var - sigma2) < 0.1
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        D = 8
        theta = torch.randn(D)
        seed = 123
        
        x1, y1, q1 = create_prompt([(theta, 5)], n_query=2, D=D, seed=seed)
        x2, y2, q2 = create_prompt([(theta, 5)], n_query=2, D=D, seed=seed)
        
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)
        assert torch.allclose(q1, q2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        D = 8
        theta = torch.randn(D)
        
        x1, y1, q1 = create_prompt([(theta, 5)], n_query=2, D=D, seed=1)
        x2, y2, q2 = create_prompt([(theta, 5)], n_query=2, D=D, seed=2)
        
        # Results should be different (very unlikely to be identical)
        assert not torch.allclose(x1, x2, atol=1e-6)
        assert not torch.allclose(y1, y2, atol=1e-6)
        assert not torch.allclose(q1, q2, atol=1e-6)
    
    def test_dimension_mismatch_theta(self):
        """Test that dimension mismatch raises error."""
        D = 8
        wrong_theta = torch.randn(10)  # Wrong dimension
        
        with pytest.raises(ValueError, match="dimension.*doesn't match"):
            create_prompt([(wrong_theta, 5)], D=D)
    
    def test_dimension_mismatch_query_theta(self):
        """Test that query_theta dimension mismatch raises error."""
        D = 8
        theta = torch.randn(D)
        wrong_query_theta = torch.randn(10)  # Wrong dimension
        
        with pytest.raises(ValueError, match="dimension.*doesn't match"):
            create_prompt(
                [(theta, 5)],
                query_theta=wrong_query_theta,
                D=D
            )
    
    def test_zero_examples(self):
        """Test with zero examples (edge case)."""
        D = 8
        theta = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta, 0)],
            n_query=2,
            D=D
        )
        
        assert x_context.shape == (0, D)
        assert y_context.shape == (0,)
        assert x_query.shape == (2, D)
    
    def test_single_query(self):
        """Test with single query point."""
        D = 8
        theta = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta, 5)],
            n_query=1,
            D=D
        )
        
        assert x_query.shape == (1, D)
    
    def test_custom_dimension(self):
        """Test with custom dimension."""
        D = 16
        theta = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta, 3)],
            n_query=2,
            D=D
        )
        
        assert x_context.shape == (3, D)
        assert y_context.shape == (3,)
        assert x_query.shape == (2, D)
    
    def test_custom_sigma2(self):
        """Test with custom noise variance."""
        D = 8
        theta = torch.randn(D)
        sigma2 = 0.5
        
        x_context, y_context, x_query = create_prompt(
            [(theta, 50)],  # More samples for better variance estimate
            n_query=1,
            D=D,
            sigma2=sigma2,
            seed=42
        )
        
        # Verify noise variance
        y_noiseless = x_context @ theta
        noise = y_context - y_noiseless
        noise_var = torch.var(noise).item()
        # With 50 samples, should be within 0.2 of target (allowing for sampling variance)
        # Sample variance can vary, especially with smaller sample sizes
        assert abs(noise_var - sigma2) < 0.2
    
    def test_output_compatible_with_predict_from_prompt(self):
        """Test that output can be used with predict_from_prompt."""
        D = 8
        theta = torch.randn(D)
        
        x_context, y_context, x_query = create_prompt(
            [(theta, 5)],
            n_query=3,
            D=D
        )
        
        # Verify types and shapes are correct for predict_from_prompt
        assert isinstance(x_context, torch.Tensor)
        assert isinstance(y_context, torch.Tensor)
        assert isinstance(x_query, torch.Tensor)
        assert x_context.dim() == 2
        assert y_context.dim() == 1
        assert x_query.dim() == 2
        assert x_context.shape[1] == D
        assert x_query.shape[1] == D
        assert x_context.shape[0] == y_context.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

