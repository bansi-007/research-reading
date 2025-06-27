"""
Week 1: Attention Is All You Need - Implementation Starter
========================================================

This file provides starter code for implementing the core components from
"Attention Is All You Need" (Vaswani et al., 2017).

Key components to implement:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Position Encoding
4. Feed Forward Network
5. Complete Transformer Block

Start with the TODO sections and gradually build up the complete implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from the original paper.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_k]
            key: [batch_size, seq_len, d_k]
            value: [batch_size, seq_len, d_v]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            attention_output: [batch_size, seq_len, d_v]
        """
        # TODO: Implement scaled dot-product attention
        # 1. Compute attention scores: Q @ K.T
        # 2. Scale by sqrt(d_k)
        # 3. Apply mask if provided
        # 4. Apply softmax
        # 5. Apply dropout
        # 6. Apply to values: attention_weights @ V
        
        d_k = query.size(-1)
        
        # Step 1 & 2: Compute scaled scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 3: Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Step 4: Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply to values
        output = torch.matmul(attention_weights, value)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Initialize linear projections for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()
        
        # TODO: Implement multi-head attention
        # 1. Apply linear transformations and reshape for multiple heads
        # 2. Apply attention for each head
        # 3. Concatenate heads
        # 4. Apply output projection
        
        # Step 1: Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 2: Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            
        attn_output = self.attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Step 4: Output projection
        output = self.w_o(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in the paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        # TODO: Implement positional encoding
        # 1. Create position indices
        # 2. Create dimension indices
        # 3. Compute angles
        # 4. Apply sin to even indices, cos to odd indices
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
            
        Returns:
            x with positional encoding added: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # TODO: Implement feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Complete Transformer block with multi-head attention and feed-forward network.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # TODO: Initialize components
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # TODO: Implement transformer block with residual connections
        # 1. Multi-head attention with residual connection and layer norm
        # 2. Feed-forward with residual connection and layer norm
        
        # Attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# Test functions
def test_attention():
    """Test the attention mechanisms with toy data."""
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test Scaled Dot-Product Attention
    attention = ScaledDotProductAttention()
    output = attention(x, x, x)
    print(f"Scaled Dot-Product Attention output shape: {output.shape}")
    
    # Test Multi-Head Attention
    multi_head_attn = MultiHeadAttention(d_model, num_heads)
    output = multi_head_attn(x, x, x)
    print(f"Multi-Head Attention output shape: {output.shape}")
    
    # Test Transformer Block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff=2048)
    output = transformer_block(x)
    print(f"Transformer Block output shape: {output.shape}")


def test_positional_encoding():
    """Test positional encoding."""
    d_model = 512
    seq_len = 100
    batch_size = 2
    
    pos_encoding = PositionalEncoding(d_model)
    x = torch.randn(seq_len, batch_size, d_model)
    output = pos_encoding(x)
    print(f"Positional encoding output shape: {output.shape}")
    
    # Visualize the positional encoding pattern
    import matplotlib.pyplot as plt
    
    pe_matrix = pos_encoding.pe[:50, 0, :50].numpy()  # First 50 positions, first 50 dimensions
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pe_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding Pattern')
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Positional encoding visualization saved as 'positional_encoding.png'")


if __name__ == "__main__":
    print("Testing Attention Mechanisms...")
    test_attention()
    
    print("\nTesting Positional Encoding...")
    test_positional_encoding()
    
    print("\nAll tests completed! You're ready to start implementing.")
    print("\nNext steps:")
    print("1. Study the paper carefully")
    print("2. Complete the TODO sections")
    print("3. Test each component thoroughly")
    print("4. Experiment with different hyperparameters")
    print("5. Try training on a simple task")
    
    # TODO for students:
    # 1. Implement a simple training loop
    # 2. Create a toy language modeling task
    # 3. Experiment with different attention patterns
    # 4. Visualize attention weights
    # 5. Compare with existing implementations 