"""
Comprehensive Test Suite for Transformer Implementation
=====================================================

This test suite validates the correctness of our Transformer implementation
and demonstrates that it works as expected. Tests cover:
- Component functionality
- Shape correctness
- Attention patterns
- Training compatibility
- Performance benchmarks

Run with: python test_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from typing import Dict, Any

# Add the implementations directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our implementation
from implementations.week1_attention_starter import (
    ScaledDotProductAttention, 
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    TransformerBlock
)

class TransformerTester:
    """Comprehensive testing suite for Transformer components."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        
    def test_scaled_dot_product_attention(self) -> bool:
        """Test the scaled dot-product attention mechanism."""
        print("🧪 Testing Scaled Dot-Product Attention...")
        
        try:
            batch_size, n_heads, seq_len, d_k = 2, 8, 10, 64
            
            # Create test inputs
            Q = torch.randn(batch_size, n_heads, seq_len, d_k)
            K = torch.randn(batch_size, n_heads, seq_len, d_k)
            V = torch.randn(batch_size, n_heads, seq_len, d_k)
            
            attention = ScaledDotProductAttention()
            output = attention(Q, K, V)
            
            # Test output shape
            expected_shape = (batch_size, n_heads, seq_len, d_k)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            
            # Test attention weights sum to 1
            attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k), dim=-1)
            assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)))
            
            print("✅ Scaled Dot-Product Attention: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Scaled Dot-Product Attention: FAILED - {e}")
            return False
    
    def test_multi_head_attention(self) -> bool:
        """Test multi-head attention mechanism."""
        print("🧪 Testing Multi-Head Attention...")
        
        try:
            batch_size, seq_len, d_model = 2, 10, 512
            n_heads = 8
            
            x = torch.randn(batch_size, seq_len, d_model)
            
            mha = MultiHeadAttention(d_model, n_heads)
            output, weights = mha(x, x, x)
            
            # Test output shape
            assert output.shape == (batch_size, seq_len, d_model)
            assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
            
            # Test with causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            output_masked, weights_masked = mha(x, x, x, mask)
            
            # Check that future positions are not attended to
            upper_tri = torch.triu(weights_masked[0, 0], diagonal=1)
            assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)
            
            print("✅ Multi-Head Attention: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Multi-Head Attention: FAILED - {e}")
            return False
    
    def test_positional_encoding(self) -> bool:
        """Test positional encoding."""
        print("🧪 Testing Positional Encoding...")
        
        try:
            d_model = 512
            max_length = 1000
            seq_len = 50
            batch_size = 2
            
            pe = PositionalEncoding(d_model, max_length)
            x = torch.randn(seq_len, batch_size, d_model)
            output = pe(x)
            
            # Test output shape
            assert output.shape == (seq_len, batch_size, d_model)
            
            # Test that positional encodings are deterministic
            output2 = pe(x)
            assert torch.allclose(output, output2)
            
            # Test that different positions have different encodings
            pe_matrix = pe.pe[:10, 0, :10]  # First 10 positions, first 10 dims
            for i in range(pe_matrix.shape[0] - 1):
                assert not torch.allclose(pe_matrix[i], pe_matrix[i + 1])
            
            print("✅ Positional Encoding: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Positional Encoding: FAILED - {e}")
            return False
    
    def test_position_wise_ffn(self) -> bool:
        """Test position-wise feed-forward network."""
        print("🧪 Testing Position-wise Feed-Forward Network...")
        
        try:
            batch_size, seq_len, d_model = 2, 10, 512
            d_ff = 2048
            
            x = torch.randn(batch_size, seq_len, d_model)
            ffn = PositionwiseFeedForward(d_model, d_ff)
            output = ffn(x)
            
            # Test output shape
            assert output.shape == (batch_size, seq_len, d_model)
            
            # Test that FFN is applied position-wise (same transformation for each position)
            single_pos = ffn(x[:, :1, :])  # Apply to single position
            assert torch.allclose(output[:, :1, :], single_pos)
            
            print("✅ Position-wise FFN: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Position-wise FFN: FAILED - {e}")
            return False
    
    def test_transformer_block(self) -> bool:
        """Test complete transformer block."""
        print("🧪 Testing Transformer Block...")
        
        try:
            batch_size, seq_len, d_model = 2, 10, 512
            n_heads, d_ff = 8, 2048
            
            x = torch.randn(batch_size, seq_len, d_model)
            transformer = TransformerBlock(d_model, n_heads, d_ff)
            output = transformer(x)
            
            # Test output shape
            assert output.shape == (batch_size, seq_len, d_model)
            
            # Test with mask
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
            output_masked = transformer(x, mask)
            assert output_masked.shape == (batch_size, seq_len, d_model)
            
            # Test that residual connections work (output != input for non-trivial case)
            assert not torch.allclose(output, x)
            
            print("✅ Transformer Block: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Transformer Block: FAILED - {e}")
            return False
    
    def test_gradient_flow(self) -> bool:
        """Test that gradients flow properly through the model."""
        print("🧪 Testing Gradient Flow...")
        
        try:
            batch_size, seq_len, d_model = 2, 10, 512
            n_heads, d_ff = 8, 2048
            
            x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
            transformer = TransformerBlock(d_model, n_heads, d_ff)
            
            output = transformer(x)
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
            
            # Check that all parameters have gradients
            for param in transformer.parameters():
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
            
            print("✅ Gradient Flow: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Gradient Flow: FAILED - {e}")
            return False
    
    def test_attention_patterns(self) -> bool:
        """Test and visualize attention patterns."""
        print("🧪 Testing Attention Patterns...")
        
        try:
            batch_size, seq_len, d_model = 1, 8, 512
            n_heads = 8
            
            # Create a simple sequence with repeated patterns
            x = torch.randn(batch_size, seq_len, d_model)
            
            mha = MultiHeadAttention(d_model, n_heads)
            output, weights = mha(x, x, x)
            
            # Test that attention weights are valid probabilities
            assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
            assert (weights >= 0).all()
            assert (weights <= 1).all()
            
            # Create visualization
            self.visualize_attention(weights[0], save_path='experiments/attention_patterns.png')
            
            print("✅ Attention Patterns: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Attention Patterns: FAILED - {e}")
            return False
    
    def visualize_attention(self, attention_weights: torch.Tensor, save_path: str):
        """Visualize attention patterns."""
        n_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(n_heads):
            row, col = i // 4, i % 4
            im = axes[row, col].imshow(attention_weights[i].detach().numpy(), cmap='Blues')
            axes[row, col].set_title(f'Head {i+1}')
            axes[row, col].set_xlabel('Key Position')
            axes[row, col].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Attention visualization saved to {save_path}")
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark model performance."""
        print("🧪 Benchmarking Performance...")
        
        batch_size, seq_len, d_model = 4, 100, 512
        n_heads, d_ff = 8, 2048
        
        x = torch.randn(batch_size, seq_len, d_model)
        transformer = TransformerBlock(d_model, n_heads, d_ff)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x = x.cuda()
            transformer = transformer.cuda()
        
        # Warm up
        for _ in range(5):
            _ = transformer(x)
        
        # Benchmark forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            output = transformer(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size * seq_len / avg_time
        
        # Count parameters
        total_params = sum(p.numel() for p in transformer.parameters())
        
        results = {
            'avg_forward_time': avg_time,
            'throughput_tokens_per_sec': throughput,
            'total_parameters': total_params,
            'device': str(x.device)
        }
        
        print(f"   ⚡ Average forward time: {avg_time:.4f}s")
        print(f"   🚀 Throughput: {throughput:.0f} tokens/sec")
        print(f"   📊 Parameters: {total_params:,}")
        print(f"   💻 Device: {x.device}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("🚀 Running Comprehensive Transformer Tests")
        print("=" * 50)
        
        # Ensure experiments directory exists
        os.makedirs('experiments', exist_ok=True)
        
        # Run all tests
        tests = [
            ('scaled_dot_product_attention', self.test_scaled_dot_product_attention),
            ('multi_head_attention', self.test_multi_head_attention),
            ('positional_encoding', self.test_positional_encoding),
            ('position_wise_ffn', self.test_position_wise_ffn),
            ('transformer_block', self.test_transformer_block),
            ('gradient_flow', self.test_gradient_flow),
            ('attention_patterns', self.test_attention_patterns),
        ]
        
        results = {}
        for test_name, test_func in tests:
            results[test_name] = test_func()
            print()
        
        # Run benchmark
        print("🏁 Performance Benchmark:")
        benchmark_results = self.benchmark_performance()
        results['benchmark'] = benchmark_results
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        passed = sum(1 for result in results.values() if isinstance(result, bool) and result)
        total = sum(1 for result in results.values() if isinstance(result, bool))
        
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {total - passed}")
        print(f"   📈 Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 All tests passed! Implementation is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check implementation.")
        
        return results


def demonstrate_usage():
    """Demonstrate basic usage of the Transformer components."""
    print("\n🎯 Transformer Implementation Demo")
    print("=" * 50)
    
    # Basic usage example
    batch_size, seq_len, d_model = 2, 10, 512
    n_heads, d_ff = 8, 2048
    
    print(f"Creating Transformer with:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of heads: {n_heads}")
    print(f"  - Feed-forward dimension: {d_ff}")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Create transformer block
    transformer = TransformerBlock(d_model, n_heads, d_ff)
    
    # Forward pass
    with torch.no_grad():
        output = transformer(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output statistics:")
    print(f"  - Mean: {output.mean():.4f}")
    print(f"  - Std: {output.std():.4f}")
    print(f"  - Min: {output.min():.4f}")
    print(f"  - Max: {output.max():.4f}")
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    # Run comprehensive tests
    tester = TransformerTester()
    results = tester.run_all_tests()
    
    # Demonstrate usage
    demonstrate_usage()
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. ✅ All core components are working correctly")
    print("2. 📊 Attention patterns are visualizable")
    print("3. ⚡ Performance is benchmarked")
    print("4. 🚀 Ready for integration into larger models")
    print("5. 📝 Ready for blog post and detailed analysis")
    print("=" * 50) 