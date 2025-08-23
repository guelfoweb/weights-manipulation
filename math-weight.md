# Mathematical foundation of weight manipulation

### Token Embedding Representation

In a transformer model, each token is represented by an embedding vector:

```
E[token_id] = [e₁, e₂, e₃, ..., e_d]
```

Where `d` is the embedding dimension (typically 768, 1024, 4096, etc.).

Let's use the token `sorry` with a simplified *8-dimensional embedding* (real models use 768+ dimensions):

#### Original Embedding

```
E["sorry"] = [0.23, -0.15, 0.67, -0.45, 0.12, 0.89, -0.34, 0.56]
```

**What the values ​​represent** 
Distributed semantic features: 

- `0.23` could encode "emotional intensity" 
- `-0.15` could represent "formality vs. informality" 
- `0.67` could indicate "negative/positive polarity" 
- `-0.45` could encode "agency/control" 
- ...

***Note:** `TinyLlama` d=2048*

```
E["sorry"] = [e₁, e₂, e₃, ..., e₂₀₄₈]
```

**Important:** 
- **These interpretations are inaccurate:** In reality, each dimension encodes a complex combination of semantic, syntactic, and contextual features that the model has learned automatically.
- **Not interpretable:** The dimensions of the embedding are distributed representations. Each semantic feature is encoded across multiple dimensions, and each dimension contributes to multiple features. There is no 1:1 mapping between dimensions and comprehensible human concepts. This is why modifying these values produces unpredictable effects.

### Weight manipulation formula
The manipulation applies a scalar factor to the entire embedding vector (`α` is the modification factor):

##### REDUCE operation

```
E'[token_id] = E[token_id] × (1 - α)
```

#### BOOST operation

```
E'[token_id] = E[token_id] × (1 + α)
```

#### ZERO operation

```
E'[token_id] = [0, 0, 0, ..., 0]
```

#### After REDUCE 0.3 operation

```
E'["sorry"] = E["sorry"] × (1 - 0.3) = E["sorry"] × 0.7

E'["sorry"] = [0.161, -0.105, 0.469, -0.315, 0.084, 0.623, -0.238, 0.392]
```

**Before modification**: The vector `[0.23, -0.15, 0.67, ...]` strongly activates apologetic semantic pathways.

**After modification**: The reduced vector `[0.161, -0.105, 0.469, ...]` has weaker activation, making the model less likely to generate `sorry` in similar contexts.

### Why modern models resist

During text generation, the modified embedding affects the initial hidden state:

```
h₀ = E'[token_id] + positional_encoding
```

This modified representation passes **through multiple transformation layers**:

```
h₁ = LayerNorm(h₀ + MultiHeadAttention(h₀))
h₂ = LayerNorm(h₁ + FeedForward(h₁))
...
hₙ = LayerNorm(hₙ₋₁ + FeedForward(hₙ₋₁))
```

The mathematical limitation becomes clear: modifying `E[token_id]` only affects the initial state `h₀`. 

The subsequent layers can mathematically reconstruct the intended semantic representation through:

- **Attention mechanisms** that weight information from other tokens
- **Feed-forward networks** that apply learned transformations
- **Layer normalization** that can compensate for altered magnitudes

The modification's impact diminishes exponentially through the layer stack, as each transformation can partially or fully recover the original semantic content through learned associations with unchanged parameters.

> This mathematical reality explains why the technique works on shallow or simple models but fails on deep, modern architectures with distributed representations.

### TinyLlama-1.1B-intermediate acrhitecture

```bash
TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T$ cat config.json 
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.55.0",
  "use_cache": true,
  "vocab_size": 32000
}
```

#### Embedding size (input/output)

```
"hidden_size": 2048
```

#### Attention heads

```
"num_attention_heads": 32
```
##### Mathematical structure

```
Input: [2048 dimensions] 
	- Split into 32 heads: ["head_dim": 64 -> dimensions per head] 
	- 32 parallel attention computations 
	- Concatenate results: [2048 total dimensions]
```
##### Calculating dimensions for head

```
num_attention_heads × head_dim = 32 × 64 = 2048
```

#### Feed-Forward Network

```
"intermediate_size": 5632
```

Size: 2048 → `5632` → 2048

##### Mathematical structure:

```
Input: [2048 size] 
	- Linear Layer 1: 2048 → 5632 
	- Activation (SiLU): specified by "hidden_act": "silu" 
	- Linear Layer 2: 5632 → 2048 
	- Output: [2048 size]
```

The feed-forward network **expands** the representation **to an intermediate dimension** for more complex processing, then compresses it back to the original dimension. This pattern (expand-process-compress) allows the model to learn complex nonlinear transformations on the data.
#### Layer blocks

```
"num_hidden_layers": 22
```

The model has `22 blocks` (or "layers") repeated one after the other. Each block is a functional unit that performs transformations on the data.

For each layer block:

- 1 `Multi-Head Attention layers`: In each block there is a multi-head attention layer: it allows the model to weight different relationships between positions in the sequence in parallel (more "heads" = more attention perspectives).
- 1 `Feed-Forward Network`: After attention, each block has a feed-forward network (typically two dense layers with a nonlinearity) that processes each position independently to increase the representational capacity.
- 2 `Layer Normalization (pre-attention e pre-FFN)`: There are two normalization operations to stabilize the training: one before the attention layer (pre-attention) and one before the FFN (pre-FFN). Therefore, each block applies the norm layer twice.

**Repeat this block 22 times in sequence**

```
Input → Embedding → [LayerNorm → Attention → LayerNorm → FFN] × 22 → Output
```

This means that when you modify the `sorry` embedding, that modification must go through 22 attention layers, 22 feed-forward layers, and 44 normalization layer operations before affecting the final output.

#### Schema

```
             input x
               |
          +----v----+
  1       | Layer   |   (LayerNorm_preAtt)
          | Norm    |
          +----+----+
               |
               v
      +---------------------+ 
      | Multi-Head Attention|   ("num_attention_heads": 32)
      +---------+-----------+ 
                |
                v
           +----+----+
           |  +res   |   (residual connection => x + Attention(x))
           +----+----+
                |
          +-----v-----+
          | Layer     |   (LayerNorm_preFFN)
  2       | Norm      |
          +-----+-----+
                |
                v
         +---------------+
         |  Feed-Forward |   (FFN: 2048 → 5632 → 2048)
         +------+--------+
                |
                v
           +----+----+
           |  +res   |   (residual connection => x + FFN(x))
           +----+----+
                |
             output

```


The modified `sorry` embedding must traverse **22 processing layers**, but the safety distribution is still concentrated primarily in the embeddings rather than distributed throughout the architecture. 

This explains the increased robustness: `+ layers == + opportunities` to reconstruct/repair modified representations.
