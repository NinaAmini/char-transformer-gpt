# 🧠 Character-Level GPT from Scratch

A small GPT-style decoder-only Transformer implemented from scratch in PyTorch and trained on the Tiny Shakespeare dataset. 

This project is primarily an educational implementation designed to understand how GPT-style language models work internally - from character tokenization and embeddings to causal self-attention, Transformer blocks, training batches, and autoregressive text generation.


---

## ✨ What This Project Does

The model learns to predict the next character given the characters that came before it. 
For example:

**Input:** `"ROMEO:"`  
**Target:** `"ROMEO:W"`

The model learns this task repeatedly across the Tiny Shakespeare corpus. Once trained, it can generate new text one character at a time:

> *ROMEO: What light through yonder window breaks...*

The generated text is not retrieved from the dataset. It is produced by the trained Transformer model.

---

## 🏗️ Architecture

The model is a compact decoder-only Transformer with:
* Character-level tokenization
* Learned token embeddings
* Learned positional embeddings
* Causal multi-head self-attention
* Feed-forward networks
* Layer normalization
* Residual connections
* Autoregressive next-character prediction

### Model Configuration

| Parameter | Value |
| --- | --- |
| Context length (`BLOCK`) | 128 |
| Batch size (`BATCH`) | 64 |
| Embedding dimension (`D_MODEL`) | 128 |
| Attention heads (`HEADS`) | 4 |
| Transformer layers (`LAYERS`) | 4 |
| Tokenization | Character-level |

*The feed-forward network expands the representation from 128 → 512 → 128.*

---

## 🔄 Model Pipeline

### The Complete Data Flow

```text
Raw Shakespeare Text
        ↓
Character Vocabulary
        ↓
Character → Integer Encoding
        ↓
Token Embedding
        +
Position Embedding
        ↓
Transformer Layer × 4
        ↓
Final LayerNorm
        ↓
Linear Output Head
        ↓
Vocabulary Logits
        ↓
Next-Character Prediction
```

### Inside each Transformer layer

```text
Input
  ↓
LayerNorm
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output
```

---

## 🔤 Character-Level Tokenization

Unlike many modern language models that use subword tokenization, this project treats each individual character as a token. 
For example, `"hello"` might become: `[17, 5, 12, 12, 15]`.

Two mappings are created:
* `stoi` (string-to-integer): Maps characters to integers (`character → integer`)
* `itos` (integer-to-string): Maps integers back to characters (`integer → character`)

The model therefore operates on numerical representations while generated predictions can be converted back into readable text.

---

## 📍 Positional Embeddings

Self-attention does not inherently know the order of tokens. For example, `"abc"` and `"cba"` contain the same characters but have completely different meanings.

To provide positional information, the model learns a separate embedding for each position in the context window. The token and position embeddings are then added together before entering the Transformer layers:

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

---

##  Causal Self-Attention

The model uses causal attention so that a character can only attend to itself and characters that came before it. 

```text
       1  2  3  4
1      ✓  ✗  ✗  ✗
2      ✓  ✓  ✗  ✗
3      ✓  ✓  ✓  ✗
4      ✓  ✓  ✓  ✓
```

This prevents the model from seeing future characters while training. Without this restriction, the model could simply look at the correct answer instead of learning to predict it.

---

##  Training Data

Training examples are created by taking random chunks from the dataset. The target sequence is shifted one character relative to the input.

**Input (x):**  `h e l l o`  
**Target (y):** `e l l o _`  

This teaches the model the fundamental language-modeling task: *Given the previous characters, predict the next character.*

A batch contains multiple randomly sampled sequences shaped as `(BATCH, BLOCK)` → `(64, 128)`.

---

##  Text Generation

After training, the model can generate text autoregressively. Starting from a seed (e.g., `"ROMEO:"`), the model:

1. Processes the current context.
2. Produces logits for every possible character.
3. Converts logits into probabilities using softmax.
4. Samples the next character.
5. Adds the character to the context.
6. Repeats the process.

**Conceptually:**
`"ROMEO:"` → predict next character → `"ROMEO:W"` → predict next character → `"ROMEO:Wh"` → ...

### Temperature
Text generation uses a `temperature` parameter to control randomness.
* **Lower temperature:** more predictable
* **Higher temperature:** more random

*The current implementation uses `temperature=0.8`.*

---

## 📁 Project Structure

```text
char-transformer-gpt/
│
├──  main.ipynb
│   └── Main code containing the model, training pipeline, and text generation.
│
├── tinyshakespeare.txt
│   └── Tiny Shakespeare training corpus.
│
├── requirements.txt
│   └── Python dependencies required to run the project.
│
├── .gitignore
│   └── Files and directories excluded from version control.
│
├── LICENSE
│   └── Project license.
│
└── README.md
    └── Project documentation.
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NinaAmini/char-transformer-gpt
   cd char-transformer-gpt
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment (macOS/Linux):**
   ```bash
   source venv/bin/activate
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📓 Running the Model

Open Jupyter Notebook, launch Jupyter and open `main.ipynb`:
```bash
jupyter notebook
```

The execution step-by-step includes:
* Downloading/loading the dataset
* Building the character vocabulary
* Encoding the text
* Creating training/validation splits
* Defining the Transformer layer & attention mechanism
* Building the GPT model
* Generating unconditioned/untrained text
* Training the model on Tiny Shakespeare
* Generating trained Shakespeare-style text

---

## 🧪 Before vs. After Training

* **Before training:** The model's parameters are randomly initialized, so generated text is essentially random gibberish.
* **After training:** The model learns statistical patterns from Shakespeare's writing and can generate text with Shakespeare-like character sequences and formatting.

*Note: The model is not expected to reproduce Shakespeare perfectly. This is a deliberately small model intended to demonstrate the mechanics of language modeling rather than achieve state-of-the-art generation quality.*

---

## 🎯 Learning Goals

This project was built to understand the fundamentals behind GPT-style language models, including:
- [x] How text is converted into numerical data
- [x] How vocabularies and token IDs work
- [x] How embeddings represent tokens
- [x] Why positional information is necessary
- [x] How self-attention works
- [x] Why causal masking is required for autoregressive generation
- [x] How residual connections are used
- [x] Why LayerNorm is used in Transformer blocks
- [x] How Transformer layers are stacked
- [x] How next-token prediction works
- [x] How logits become probabilities
- [x] How autoregressive generation works
- [x] How temperature affects sampling

---

## 📚 References & Inspiration

This project is inspired by the educational approach of building language models from the ground up, particularly the work of Andrej Karpathy on small GPT implementations.

* Karpathy's minGPT
* Tiny Shakespeare dataset
* PyTorch documentation

---

## 📄 License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.

---

## 👩‍💻 Author

**Nina Amini**  
GitHub: [@NinaAmini](https://github.com/NinaAmini)