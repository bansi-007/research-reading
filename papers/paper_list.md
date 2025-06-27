# Essential LLM Papers - Detailed List

## Phase 1: Foundations (Weeks 1-3)

### 1. "Attention Is All You Need" (2017)
- **Authors**: Ashish Vaswani, Noam Shazeer, et al.
- **Venue**: NeurIPS 2017
- **ArXiv**: https://arxiv.org/abs/1706.03762
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Introduces the Transformer architecture
  - Self-attention mechanism
  - Positional encoding
  - Multi-head attention
- **Why Essential**: Foundation of all modern LLMs
- **Implementation Focus**: Multi-head attention, positional encoding

### 2. "Deep Residual Learning for Image Recognition" (2015)
- **Authors**: Kaiming He, Xiangyu Zhang, et al.
- **Venue**: CVPR 2016
- **ArXiv**: https://arxiv.org/abs/1512.03385
- **Difficulty**: ⭐⭐
- **Key Contributions**:
  - Residual connections
  - Deep network training techniques
- **Why Essential**: Residual connections are crucial in Transformers
- **Implementation Focus**: Skip connections, deep network initialization

### 3. "Layer Normalization" (2016)
- **Authors**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
- **ArXiv**: https://arxiv.org/abs/1607.06450
- **Difficulty**: ⭐⭐
- **Key Contributions**:
  - Layer normalization technique
  - Comparison with batch normalization
- **Why Essential**: Standard normalization in Transformers
- **Implementation Focus**: LayerNorm implementation and variants

---

## Phase 2: Pre-training Era (Weeks 4-6)

### 4. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- **Authors**: Jacob Devlin, Ming-Wei Chang, et al.
- **ArXiv**: https://arxiv.org/abs/1810.04805
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Bidirectional context modeling
  - Masked Language Modeling (MLM)
  - Next Sentence Prediction (NSP)
  - Transfer learning paradigm
- **Why Essential**: Revolutionized NLP with pre-training + fine-tuning
- **Implementation Focus**: MLM training, BERT architecture

### 5. "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
- **Authors**: Alec Radford, Karthik Narasimhan, et al.
- **OpenAI**: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Unsupervised pre-training for language understanding
  - Autoregressive language modeling
  - Task-specific fine-tuning
- **Why Essential**: Established the GPT paradigm
- **Implementation Focus**: Decoder-only Transformer, autoregressive training

### 6. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- **Authors**: Alec Radford, Jeffrey Wu, et al.
- **OpenAI**: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Zero-shot task transfer
  - Scaling benefits
  - WebText dataset
- **Why Essential**: Demonstrated emergent capabilities from scale
- **Implementation Focus**: GPT-2 architecture, zero-shot evaluation

---

## Phase 3: Scaling Laws (Weeks 7-8)

### 7. "Language Models are Few-Shot Learners" (GPT-3, 2020)
- **Authors**: Tom B. Brown, Benjamin Mann, et al.
- **ArXiv**: https://arxiv.org/abs/2005.14165
- **Difficulty**: ⭐⭐⭐⭐
- **Key Contributions**:
  - In-context learning
  - Few-shot prompting
  - Emergence at scale (175B parameters)
  - Comprehensive evaluation
- **Why Essential**: Paradigm shift to prompting-based interaction
- **Implementation Focus**: In-context learning, prompting strategies

### 8. "Scaling Laws for Neural Language Models" (2020)
- **Authors**: Jared Kaplan, Sam McCandlish, et al.
- **ArXiv**: https://arxiv.org/abs/2001.08361
- **Difficulty**: ⭐⭐⭐⭐
- **Key Contributions**:
  - Power law relationships for model performance
  - Optimal compute allocation
  - Data vs. model size tradeoffs
- **Why Essential**: Mathematical foundation for LLM scaling
- **Implementation Focus**: Scaling experiments, power law fitting

---

## Phase 4: Alignment and Safety (Weeks 9-10)

### 9. "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- **Authors**: Long Ouyang, Jeff Wu, et al.
- **ArXiv**: https://arxiv.org/abs/2203.02155
- **Difficulty**: ⭐⭐⭐⭐⭐
- **Key Contributions**:
  - Reinforcement Learning from Human Feedback (RLHF)
  - Instruction following
  - Human preference modeling
- **Why Essential**: Foundation of ChatGPT and instruction-following models
- **Implementation Focus**: RLHF pipeline, reward modeling

### 10. "Constitutional AI: Harmlessness from AI Feedback" (2022)
- **Authors**: Yuntao Bai, Andy Jones, et al.
- **ArXiv**: https://arxiv.org/abs/2212.08073
- **Difficulty**: ⭐⭐⭐⭐
- **Key Contributions**:
  - Self-supervision for AI safety
  - Constitutional principles
  - AI feedback instead of human feedback
- **Why Essential**: Alternative approach to alignment
- **Implementation Focus**: Constitutional training, self-critique

---

## Phase 5: Efficiency and Practical Techniques (Weeks 11-12)

### 11. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Authors**: Edward J. Hu, Yelong Shen, et al.
- **ArXiv**: https://arxiv.org/abs/2106.09685
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Parameter-efficient fine-tuning
  - Low-rank matrix decomposition
  - Minimal performance degradation
- **Why Essential**: Practical fine-tuning for large models
- **Implementation Focus**: LoRA adapters, rank selection

### 12. "LLaMA: Open and Efficient Foundation Language Models" (2023)
- **Authors**: Hugo Touvron, Thibaut Lavril, et al.
- **ArXiv**: https://arxiv.org/abs/2302.13971
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Efficient architecture choices
  - Strong performance with fewer parameters
  - Open research approach
- **Why Essential**: State-of-the-art efficient architecture
- **Implementation Focus**: RMSNorm, SwiGLU, RoPE

---

## Bonus Papers (Advanced Topics)

### 13. "Chain-of-Thought Prompting Elicits Reasoning" (2022)
- **Authors**: Jason Wei, Xuezhi Wang, et al.
- **ArXiv**: https://arxiv.org/abs/2201.11903
- **Difficulty**: ⭐⭐⭐
- **Key Contributions**:
  - Reasoning through intermediate steps
  - Emergent ability in large models
  - Few-shot chain-of-thought
- **Why Essential**: Unlocks reasoning capabilities
- **Implementation Focus**: CoT prompting, reasoning evaluation

### 14. "PaLM: Scaling Language Modeling with Pathways" (2022)
- **Authors**: Aakanksha Chowdhery, Sharan Narang, et al.
- **ArXiv**: https://arxiv.org/abs/2204.02311
- **Difficulty**: ⭐⭐⭐⭐
- **Key Contributions**:
  - 540B parameter model
  - Pathways system for efficient training
  - Breakthrough capabilities
- **Why Essential**: Demonstrates extreme scale benefits
- **Implementation Focus**: Efficient large-scale training

### 15. "Sparks of Artificial General Intelligence: Early look at GPT-4" (2023)
- **Authors**: Sébastien Bubeck, Varun Chandrasekaran, et al.
- **ArXiv**: https://arxiv.org/abs/2303.12712
- **Difficulty**: ⭐⭐⭐⭐
- **Key Contributions**:
  - Comprehensive evaluation of GPT-4
  - AGI-like capabilities analysis
  - Novel evaluation methodologies
- **Why Essential**: Understanding frontier model capabilities
- **Implementation Focus**: Evaluation frameworks, capability assessment

---

## Legend
- ⭐ = Beginner friendly
- ⭐⭐ = Requires basic ML knowledge
- ⭐⭐⭐ = Intermediate - requires Transformer familiarity
- ⭐⭐⭐⭐ = Advanced - complex concepts
- ⭐⭐⭐⭐⭐ = Expert level - cutting-edge research

## Quick Access Links
- [ArXiv Sanity](http://arxiv-sanity.com/) - For paper discovery
- [Papers With Code](https://paperswithcode.com/) - For implementations
- [Hugging Face Papers](https://huggingface.co/papers) - For recent papers 