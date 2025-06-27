"""
Attention Is All You Need - Complete Implementation for Beginners
================================================================

This file contains a complete implementation of the Transformer architecture
from the paper "Attention Is All You Need" (Vaswani et al., 2017).

Every line is commented to help beginners understand how the mathematical
concepts translate to working code.

Author: Research Paper Study
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# ============================================================================
# PART 1: FUNDAMENTAL BUILDING BLOCKS
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Adds position information to word embeddings using sine and cosine functions.
    
    Why we need this:
    - Attention mechanism has no sense of word order
    - "Cat sat on mat" vs "Mat sat on cat" would be identical without position info
    - This encoding gives each position a unique mathematical fingerprint
    
    Mathematical Intuition:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    - Different frequencies for different dimensions
    - Creates unique patterns that the model can learn to use
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model (usually 512)
            max_len: Maximum sequence length we'll support
        """
        super().__init__()
        
        # Create a matrix to hold positional encodings
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sine/cosine functions
        # This creates different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
        
        Returns:
            Embeddings with positional information added
        """
        # Add positional encoding to input
        # x.size(0) is the sequence length
        return x + self.pe[:x.size(0), :]

class ScaledDotProductAttention(nn.Module):
    """
    The core attention mechanism that determines how much each word should
    focus on every other word in the sequence.
    
    Mathematical Formula:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Intuitive Explanation:
    1. Q (Query): "What am I looking for?"
    2. K (Key): "What information do I have?"
    3. V (Value): "What is the actual content?"
    4. QK^T: Similarity between queries and keys
    5. √d_k: Scaling factor to prevent gradients from vanishing
    6. softmax: Convert similarities to probabilities (sum to 1)
    7. Multiply by V: Get weighted average of values
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_k]
            key: Key tensor [batch_size, seq_len, d_k]
            value: Value tensor [batch_size, seq_len, d_v]
            mask: Optional mask to ignore certain positions
        
        Returns:
            output: Attention output [batch_size, seq_len, d_v]
            attention_weights: Attention weights [batch_size, seq_len, seq_len]
        """
        d_k = query.size(-1)  # Dimension of key vectors
        
        # Step 1: Calculate similarity scores
        # MatMul between Query and Key^T gives us similarity matrix
        # Shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by √d_k to prevent vanishing gradients
        # Without this, for large d_k, softmax becomes too sharp
        # Mathematical intuition: dot products grow with √d_k for random vectors
        scores = scores / math.sqrt(d_k)
        
        # Step 3: Apply mask if provided
        # Typically used to prevent attention to padding tokens or future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention probabilities
        # Each row sums to 1.0 - this is the attention distribution
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply attention to values
        # This creates weighted average of all values based on attention weights
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Why Multiple Heads?
    - Single attention head can only capture one type of relationship
    - Different heads can specialize in different patterns:
      * Head 1: Syntactic relationships (subject-verb)
      * Head 2: Semantic relationships (meaning)
      * Head 3: Positional relationships (nearby words)
      * Head 4: Reference relationships (pronouns)
    
    Mathematical Process:
    1. Split Q, K, V into multiple heads
    2. Apply attention to each head independently
    3. Concatenate results from all heads
    4. Apply final linear transformation
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (usually 512)
            num_heads: Number of attention heads (usually 8)
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear layers to create Q, K, V for all heads at once
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask
        
        Returns:
            output: Multi-head attention output [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Step 1: Apply linear transformations and split into heads
        # Transform input to Q, K, V and reshape for multiple heads
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Now shapes are: [batch_size, num_heads, seq_len, d_k]
        
        # Step 2: Apply attention to each head
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads
        # Transpose back and concatenate: [batch_size, seq_len, num_heads * d_k]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Step 4: Apply final linear transformation
        output = self.W_o(attention_output)
        
        return self.dropout(output)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network applied to each position separately.
    
    Why do we need this?
    - Attention is just weighted averages (linear operations)
    - FFN adds non-linearity and allows complex transformations
    - Applied to each position independently (no interaction between positions)
    
    Architecture:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Think of it as:
    1. Expand to higher dimension (usually 4x larger)
    2. Apply ReLU activation (non-linearity)
    3. Project back to original dimension
    
    This is like having a mini neural network at each word position.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (input/output size)
            d_ff: Feed-forward dimension (hidden size, usually 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Two linear transformations with ReLU in between
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
        """
        # First linear transformation + ReLU + dropout
        output = self.dropout(F.relu(self.linear1(x)))
        
        # Second linear transformation
        return self.linear2(output)


# ============================================================================
# PART 2: TRANSFORMER BLOCKS
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer consisting of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm (residual connection + layer normalization)
    
    Residual Connections (Add & Norm):
    - Helps with gradient flow in deep networks
    - Like having express lanes in a skyscraper
    - output = LayerNorm(input + sublayer(input))
    
    Layer Normalization:
    - Normalizes across the feature dimension
    - Keeps values in a reasonable range
    - Speeds up training and improves stability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            output: Encoded tensor [batch_size, seq_len, d_model]
        """
        # Step 1: Multi-head self-attention with residual connection and layer norm
        # Self-attention: query = key = value = x
        attention_output = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Step 2: Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder consisting of:
    1. Input embeddings + positional encoding
    2. Stack of N encoder layers
    3. Optional final layer normalization
    
    The encoder processes the input sequence and creates rich representations
    that capture both local and global dependencies.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layer converts token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding adds position information
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            src: Source sequence [batch_size, seq_len]
            mask: Optional attention mask
        
        Returns:
            output: Encoded representations [batch_size, seq_len, d_model]
        """
        # Step 1: Convert tokens to embeddings and scale
        # Scaling by √d_model is mentioned in the paper
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Step 2: Add positional encoding
        x = x.transpose(0, 1)  # Change to [seq_len, batch_size, d_model] for pos encoding
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Change back to [batch_size, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Step 3: Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Step 4: Final layer normalization
        return self.norm(x)


# ============================================================================
# PART 3: COMPLETE TRANSFORMER MODEL
# ============================================================================

class SimpleTransformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    This is a simplified version focusing on the encoder part.
    
    For tasks like:
    - Sentiment analysis
    - Text classification
    - Feature extraction
    
    The full encoder-decoder architecture would be used for:
    - Machine translation
    - Text summarization
    - Question answering
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_len: int = 5000,
                 num_classes: int = 2, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (512 in original paper)
            num_heads: Number of attention heads (8 in original paper)
            num_layers: Number of encoder layers (6 in original paper)
            d_ff: Feed-forward dimension (2048 in original paper)
            max_len: Maximum sequence length
            num_classes: Number of output classes (for classification tasks)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the complete model.
        
        Args:
            src: Source sequence [batch_size, seq_len]
            mask: Optional attention mask
        
        Returns:
            output: Classification logits [batch_size, num_classes]
        """
        # Encode the sequence
        encoded = self.encoder(src, mask)
        
        # Use the first token's representation for classification (like BERT's [CLS])
        cls_representation = encoded[:, 0, :]  # [batch_size, d_model]
        
        # Apply classification head
        output = self.classifier(cls_representation)
        
        return output


# ============================================================================
# PART 4: UTILITY FUNCTIONS FOR UNDERSTANDING
# ============================================================================

def visualize_attention_weights(attention_weights: torch.Tensor, tokens: list, 
                              head_idx: int = 0, layer_idx: int = 0):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens/words
        head_idx: Which attention head to visualize
        layer_idx: Which layer to visualize
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract weights for specific head and layer
    weights = attention_weights[0, head_idx].detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                annot=True, 
                fmt='.2f',
                cmap='Blues')
    plt.title(f'Attention Weights - Head {head_idx}, Layer {layer_idx}')
    plt.xlabel('Keys (what we attend to)')
    plt.ylabel('Queries (what we are)')
    plt.show()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_padding_mask(sequences: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask to ignore padding tokens in attention.
    
    Args:
        sequences: Input sequences [batch_size, seq_len]
        pad_token_id: ID of padding token
    
    Returns:
        mask: Padding mask [batch_size, 1, 1, seq_len]
    """
    mask = (sequences != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


# ============================================================================
# PART 5: EXAMPLE USAGE AND TESTING
# ============================================================================

def test_transformer_components():
    """Test individual components of the Transformer."""
    print("🧪 Testing Transformer Components\n")
    
    # Test parameters
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test 1: Positional Encoding
    print("1️⃣ Testing Positional Encoding...")
    pe = PositionalEncoding(d_model)
    x_with_pos = pe(x.transpose(0, 1)).transpose(0, 1)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_with_pos.shape}")
    print(f"   ✅ Positional encoding working!\n")
    
    # Test 2: Scaled Dot-Product Attention
    print("2️⃣ Testing Scaled Dot-Product Attention...")
    attention = ScaledDotProductAttention()
    attn_output, attn_weights = attention(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {attn_output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   Attention weights sum (should be ~1.0): {attn_weights[0, 0].sum():.3f}")
    print(f"   ✅ Attention working!\n")
    
    # Test 3: Multi-Head Attention
    print("3️⃣ Testing Multi-Head Attention...")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {mha_output.shape}")
    print(f"   Number of parameters: {count_parameters(mha):,}")
    print(f"   ✅ Multi-head attention working!\n")
    
    # Test 4: Feed-Forward Network
    print("4️⃣ Testing Feed-Forward Network...")
    ffn = PositionwiseFeedForward(d_model, d_model * 4)
    ffn_output = ffn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {ffn_output.shape}")
    print(f"   ✅ Feed-forward network working!\n")
    
    # Test 5: Complete Encoder Layer
    print("5️⃣ Testing Encoder Layer...")
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_model * 4)
    layer_output = encoder_layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {layer_output.shape}")
    print(f"   Number of parameters: {count_parameters(encoder_layer):,}")
    print(f"   ✅ Encoder layer working!\n")
    
    print("🎉 All tests passed! The Transformer is ready to use.")


def demo_simple_transformer():
    """Demonstrate the complete Transformer model."""
    print("🚀 Demonstrating Complete Transformer Model\n")
    
    # Model parameters (smaller for demo)
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    num_classes = 2
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=num_classes
    )
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {count_parameters(model):,}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of heads: {num_heads}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Feed-forward dimension: {d_ff}")
    
    # Demo input
    batch_size, seq_len = 4, 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n📝 Input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"\n📤 Output:")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Sample predictions: {torch.softmax(outputs[0], dim=-1)}")
    
    print(f"\n✅ Model is working correctly!")


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 ATTENTION IS ALL YOU NEED - IMPLEMENTATION")
    print("=" * 70)
    print()
    
    # Run tests
    test_transformer_components()
    print("\n" + "=" * 70 + "\n")
    demo_simple_transformer()
    
    print("\n" + "=" * 70)
    print("🎓 Ready to revolutionize AI with Transformers!")
    print("=" * 70) 