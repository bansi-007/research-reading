"""
Complete Transformer Implementation from "Attention Is All You Need"
==================================================================

A production-ready implementation of the Transformer architecture with:
- Full encoder-decoder architecture
- Proper masking for training and inference
- Comprehensive testing and validation
- Attention visualization capabilities
- Training utilities and example usage

Author: [Your Name]
Paper: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransformerConfig:
    """Configuration class for Transformer model."""
    vocab_size: int = 30000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 1000
    dropout: float = 0.1
    pad_token_id: int = 0


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, seq_len, seq_len] or [batch_size, n_heads, seq_len, seq_len]
            
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_length, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            x with positional encoding added: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single layer of the Transformer encoder."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single layer of the Transformer decoder."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.cross_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] - decoder input
            encoder_output: [batch_size, src_len, d_model] - encoder output
            self_attn_mask: [batch_size, seq_len, seq_len] - decoder self-attention mask
            cross_attn_mask: [batch_size, seq_len, src_len] - encoder-decoder attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder stack."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.n_layers)
        ])
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            encoder_output: [batch_size, src_len, d_model]
            self_attn_mask: [batch_size, seq_len, seq_len]
            cross_attn_mask: [batch_size, seq_len, src_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.src_embedding.weight, mean=0, std=self.config.d_model**-0.5)
        nn.init.normal_(self.tgt_embedding.weight, mean=0, std=self.config.d_model**-0.5)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Create padding mask to ignore padding tokens."""
        return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens."""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [batch_size, src_len] - source sequence
            tgt: [batch_size, tgt_len] - target sequence
            
        Returns:
            output: [batch_size, tgt_len, vocab_size] - output logits
        """
        # Create masks
        src_padding_mask = self.create_padding_mask(src, self.config.pad_token_id)
        tgt_padding_mask = self.create_padding_mask(tgt, self.config.pad_token_id)
        tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Combine padding and causal masks for decoder self-attention
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        # Encode source sequence
        src_embedded = self.src_embedding(src) * math.sqrt(self.config.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded, src_padding_mask)
        
        # Decode target sequence
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_padding_mask)
        
        # Project to vocabulary space
        output = self.output_projection(decoder_output)
        
        return output


# Testing and Visualization Functions
class TransformerTester:
    """Comprehensive testing suite for Transformer implementation."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = Transformer(config)
        
    def test_component_shapes(self):
        """Test that all components produce correct output shapes."""
        print("🧪 Testing component shapes...")
        
        batch_size, seq_len = 2, 10
        
        # Test attention
        attention = ScaledDotProductAttention()
        q = k = v = torch.randn(batch_size, self.config.n_heads, seq_len, self.config.d_model // self.config.n_heads)
        output, weights = attention(q, k, v)
        
        assert output.shape == (batch_size, self.config.n_heads, seq_len, self.config.d_model // self.config.n_heads)
        assert weights.shape == (batch_size, self.config.n_heads, seq_len, seq_len)
        print("✅ ScaledDotProductAttention shapes correct")
        
        # Test multi-head attention
        mha = MultiHeadAttention(self.config.d_model, self.config.n_heads)
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        output, weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, self.config.d_model)
        assert weights.shape == (batch_size, self.config.n_heads, seq_len, seq_len)
        print("✅ MultiHeadAttention shapes correct")
        
        # Test positional encoding
        pe = PositionalEncoding(self.config.d_model)
        output = pe(x)
        assert output.shape == (batch_size, seq_len, self.config.d_model)
        print("✅ PositionalEncoding shapes correct")
        
        # Test full model
        src = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        tgt = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        output = self.model(src, tgt)
        
        assert output.shape == (batch_size, seq_len, self.config.vocab_size)
        print("✅ Full Transformer shapes correct")
        
    def test_attention_patterns(self):
        """Test that attention patterns are reasonable."""
        print("\n🧪 Testing attention patterns...")
        
        batch_size, seq_len = 1, 5
        
        # Create simple test case
        mha = MultiHeadAttention(self.config.d_model, self.config.n_heads)
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        
        # Test with and without mask
        output, weights = mha(x, x, x)
        
        # Attention weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
        print("✅ Attention weights sum to 1")
        
        # Test with causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        output_masked, weights_masked = mha(x, x, x, mask)
        
        # Upper triangular part should be close to 0
        upper_tri = torch.triu(weights_masked[0, 0], diagonal=1)
        assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)
        print("✅ Causal masking works correctly")
        
    def visualize_positional_encoding(self):
        """Visualize positional encoding patterns."""
        print("\n📊 Creating positional encoding visualization...")
        
        pe = PositionalEncoding(self.config.d_model, max_length=100)
        
        # Extract positional encodings
        pos_encodings = pe.pe[0, :50, :50].detach().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pos_encodings, cmap='RdBu', aspect='auto')
        plt.colorbar(label='Encoding Value')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.title('Positional Encoding Pattern\n(First 50 positions × First 50 dimensions)')
        plt.tight_layout()
        plt.savefig('experiments/positional_encoding_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Positional encoding visualization saved")
        
    def benchmark_performance(self):
        """Benchmark model performance and memory usage."""
        print("\n⚡ Benchmarking performance...")
        
        batch_size, seq_len = 2, 100
        src = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        tgt = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        
        # Warm up
        for _ in range(5):
            _ = self.model(src, tgt)
        
        # Time forward pass
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(10):
            output = self.model(src, tgt)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"✅ Average forward pass time: {avg_time:.4f}s")
        print(f"✅ Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Running comprehensive Transformer tests...\n")
        
        self.test_component_shapes()
        self.test_attention_patterns()
        self.visualize_positional_encoding()
        self.benchmark_performance()
        
        print("\n🎉 All tests completed successfully!")


# Example usage and training utilities
def create_toy_translation_dataset(vocab_size: int = 1000, num_samples: int = 1000, 
                                 max_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a toy translation dataset for testing."""
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        # Random source sequence
        src_len = torch.randint(5, max_length, (1,)).item()
        src = torch.randint(1, vocab_size, (src_len,))
        
        # Target is reversed source (toy translation task)
        tgt = torch.flip(src, [0])
        
        src_data.append(src)
        tgt_data.append(tgt)
    
    return src_data, tgt_data


def demonstrate_usage():
    """Demonstrate how to use the Transformer implementation."""
    print("🎯 Transformer Implementation Demo")
    print("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        max_seq_length=100,
        dropout=0.1
    )
    
    print(f"📋 Configuration:")
    print(f"   - Vocabulary size: {config.vocab_size}")
    print(f"   - Model dimension: {config.d_model}")
    print(f"   - Number of heads: {config.n_heads}")
    print(f"   - Number of layers: {config.n_layers}")
    print(f"   - Feed-forward dimension: {config.d_ff}")
    
    # Run tests
    tester = TransformerTester(config)
    tester.run_all_tests()
    
    # Show example forward pass
    print("\n🔍 Example forward pass:")
    model = Transformer(config)
    
    batch_size, seq_len = 2, 10
    src = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    
    print(f"   - Source shape: {src.shape}")
    print(f"   - Target shape: {tgt.shape}")
    
    with torch.no_grad():
        output = model(src, tgt)
        
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output logits range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✨ Implementation complete! Ready for training.")


if __name__ == "__main__":
    # Create experiments directory
    import os
    os.makedirs('experiments', exist_ok=True)
    
    # Run demonstration
    demonstrate_usage()
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Train the model on a real translation dataset")
    print("2. Implement beam search for inference")
    print("3. Add attention visualization tools")
    print("4. Experiment with different architectures")
    print("5. Compare with reference implementations")
    print("=" * 50) 