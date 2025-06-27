# Visual Diagram Templates for Medium Blog 🎨

## 1. **Attention Mechanism Flow**

```
Input Sentence: "The cat sat on the mat"

Step 1: Word Embeddings
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ The │ │ cat │ │ sat │ │ on  │ │ the │ │ mat │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
   ↓       ↓       ↓       ↓       ↓       ↓
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ 0.2 │ │ 0.8 │ │ 0.3 │ │ 0.1 │ │ 0.2 │ │ 0.4 │
│ 0.5 │ │ 0.1 │ │ 0.9 │ │ 0.6 │ │ 0.5 │ │ 0.7 │
│ 0.1 │ │ 0.6 │ │ 0.2 │ │ 0.3 │ │ 0.1 │ │ 0.3 │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘

Step 2: Attention Calculation (Focus on "sat")
     Query: "sat" = [0.3, 0.9, 0.2]
     
     Similarity with each word:
     ┌─────────────────────────────────────┐
     │ Word  │ Similarity │ Attention (%) │
     ├───────┼────────────┼───────────────┤
     │ The   │    0.63    │     28%       │
     │ cat   │    0.85    │     42%       │ ← Most important!
     │ sat   │    1.00    │     15%       │
     │ on    │    0.32    │      8%       │
     │ the   │    0.25    │      5%       │
     │ mat   │    0.18    │      2%       │
     └─────────────────────────────────────┘

Step 3: Weighted Output
Final "sat" representation = 
   0.28 × The + 0.42 × cat + 0.15 × sat + 0.08 × on + 0.05 × the + 0.02 × mat
```

## 2. **Transformer Architecture (Simplified)**

```
                    TRANSFORMER ENCODER
                    
Input: "The cat sat"
         ↓
    ┌─────────────────────────────────────────┐
    │           Word Embeddings                │
    │    ┌─────┐ ┌─────┐ ┌─────┐              │
    │    │ The │ │ cat │ │ sat │              │
    │    └─────┘ └─────┘ └─────┘              │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │        Positional Encoding              │
    │         + + +                           │
    │    ┌─────┐ ┌─────┐ ┌─────┐              │
    │    │Pos 0│ │Pos 1│ │Pos 2│              │
    │    └─────┘ └─────┘ └─────┘              │
    └─────────────────────────────────────────┘
         ↓
    ╔═════════════════════════════════════════╗
    ║           ENCODER LAYER 1               ║
    ║                                         ║
    ║  ┌─────────────────────────────────────┐ ║
    ║  │      Multi-Head Attention          │ ║
    ║  │                                   │ ║
    ║  │  Head1  Head2  Head3  ... Head8   │ ║
    ║  │    ↘      ↘      ↘        ↘      │ ║
    ║  │      ┌─────────────────────┐      │ ║
    ║  │      │    Concatenate      │      │ ║
    ║  │      └─────────────────────┘      │ ║
    ║  └─────────────────────────────────────┘ ║
    ║              ↓                          ║
    ║  ┌─────────────────────────────────────┐ ║
    ║  │         Add & Norm                 │ ║
    ║  │    Input + Attention Output        │ ║
    ║  └─────────────────────────────────────┘ ║
    ║              ↓                          ║
    ║  ┌─────────────────────────────────────┐ ║
    ║  │      Feed Forward Network          │ ║
    ║  │                                   │ ║
    ║  │   Linear → ReLU → Linear          │ ║
    ║  └─────────────────────────────────────┘ ║
    ║              ↓                          ║
    ║  ┌─────────────────────────────────────┐ ║
    ║  │         Add & Norm                 │ ║
    ║  │      Input + FFN Output            │ ║
    ║  └─────────────────────────────────────┘ ║
    ╚═════════════════════════════════════════╝
         ↓
    ... (5 more similar layers)
         ↓
    ┌─────────────────────────────────────────┐
    │              OUTPUT                     │
    │         Rich Representations           │
    │    ┌─────┐ ┌─────┐ ┌─────┐              │
    │    │ The │ │ cat │ │ sat │              │
    │    └─────┘ └─────┘ └─────┘              │
    └─────────────────────────────────────────┘
```

## 3. **Before vs After Transformers**

```
BEFORE TRANSFORMERS (RNNs)
Sequential Processing - Slow & Limited Memory

"The cat sat on the mat"

Step 1: Process "The"
┌─────┐
│ The │ → Hidden State 1
└─────┘

Step 2: Process "cat" 
┌─────┐ ┌─────┐
│ The │ │ cat │ → Hidden State 2 (forgets some of "The")
└─────┘ └─────┘

Step 3: Process "sat"
┌─────┐ ┌─────┐ ┌─────┐
│ The │ │ cat │ │ sat │ → Hidden State 3 (forgets more)
└─────┘ └─────┘ └─────┘

... by the end, "The" is mostly forgotten!

═══════════════════════════════════════════════════════════

AFTER TRANSFORMERS
Parallel Processing - Fast & Full Memory

"The cat sat on the mat"

All words processed simultaneously:
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ The │ │ cat │ │ sat │ │ on  │ │ the │ │ mat │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
   ↕       ↕       ↕       ↕       ↕       ↕
   └───────┼───────┼───────┼───────┼───────┘
           └───────┼───────┼───────┼───────┐
                   └───────┼───────┼───────┤
                           └───────┼───────┤
                                   └───────┘

Every word can attend to every other word!
All information preserved and accessible.
```

## 4. **Multi-Head Attention Specialization**

```
INPUT: "The programmer debugged the complex algorithm"

HEAD 1 - SYNTACTIC RELATIONSHIPS (Grammar)
Strong connections:
The ←→ programmer (determiner-noun)
programmer ←→ debugged (subject-verb)  
the ←→ algorithm (determiner-noun)

HEAD 2 - SEMANTIC RELATIONSHIPS (Meaning)
Strong connections:
programmer ←→ debugged (who does what)
debugged ←→ algorithm (what is debugged)
complex ←→ algorithm (what is complex)

HEAD 3 - POSITIONAL RELATIONSHIPS (Distance)
Strong connections between adjacent words:
The ←→ programmer
programmer ←→ debugged
debugged ←→ the
the ←→ complex
complex ←→ algorithm

HEAD 4 - REFERENCE RELATIONSHIPS (Pronouns/References)
In this case, no pronouns, but would connect:
- Pronouns to their referents
- Coreference resolution
```

## 5. **Attention Heatmap Visualization**

```
ATTENTION WEIGHTS MATRIX
(Darker = Higher Attention)

Sentence: "The cat sat on the mat"

           T  c  s  o  t  m
           h  a  a  n  h  a
           e  t  t     e  t

The     [■■■□□□□□□□□□□□□□□□]  ← "The" pays most attention to itself
        
cat     [□□■■■■■■□□□□□□□□□□]  ← "cat" focuses on itself and "sat"
        
sat     [□□■■■■■■■■■■□□□□□□]  ← "sat" strongly attends to "cat"
        
on      [□□□□□□■■■■■■■■□□□□]  ← "on" connects to "sat" and "mat"
        
the     [■■■□□□□□□□□□■■■■■■]  ← Second "the" connects to first "The"
        
mat     [□□□□□□□□□□■■■■■■■■]  ← "mat" focuses on "on" and itself

Legend:
■■■■ = High attention (0.7-1.0)
■■■  = Medium attention (0.4-0.7)  
■■   = Low attention (0.1-0.4)
□    = Minimal attention (0.0-0.1)
```

## 6. **Mathematical Intuition Visual**

```
ATTENTION CALCULATION STEP-BY-STEP

Given: Query="sat", Keys=["The","cat","sat","on","the","mat"]

Step 1: Dot Product (Similarity)
┌─────────────────────────────────────────────────────────┐
│ Query: [0.3, 0.9, 0.2]                                 │
│                                                         │
│ Keys:                      Dot Products:                │
│ "The": [0.1, 0.3, 0.4] →  0.3×0.1 + 0.9×0.3 + 0.2×0.4 │
│                         →  0.03 + 0.27 + 0.08 = 0.38   │
│                                                         │
│ "cat": [0.8, 0.1, 0.6] →  0.3×0.8 + 0.9×0.1 + 0.2×0.6 │
│                         →  0.24 + 0.09 + 0.12 = 0.45   │
│                                                         │
│ "sat": [0.3, 0.9, 0.2] →  0.3×0.3 + 0.9×0.9 + 0.2×0.2 │
│                         →  0.09 + 0.81 + 0.04 = 0.94   │
└─────────────────────────────────────────────────────────┘

Step 2: Scale by √d_k
d_k = 3, so √d_k = 1.73
Scaled scores: [0.38/1.73, 0.45/1.73, 0.94/1.73]
              = [0.22, 0.26, 0.54]

Step 3: Softmax (Convert to Probabilities)
exp(0.22) = 1.25, exp(0.26) = 1.30, exp(0.54) = 1.72
Sum = 1.25 + 1.30 + 1.72 = 4.27

Probabilities: [1.25/4.27, 1.30/4.27, 1.72/4.27]
              = [0.29, 0.30, 0.40]
              = [29%, 30%, 40%]

Step 4: Weighted Sum with Values
Final output = 0.29×Value("The") + 0.30×Value("cat") + 0.40×Value("sat")
```
