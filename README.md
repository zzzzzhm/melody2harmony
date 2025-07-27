# 🎵 Symbolic Music Generation 

This repository showcases symbolic music generation using different techniques:

---

### 1️⃣ Symbolic Unconditioned Generation (MiniLLaMA Transformer)

- Learn the distribution over symbolic music sequences and generate melody samples.
- **Model**: MiniLLaMA Transformer pretrained on REMI-style tokens.
- **Dataset**: Nottingham symbolic melody dataset
- **Baseline**: n-gram Markov chain for comparison
- **Highlights**:
  - Uses REMI (REvamped MIDI) tokenization
  - Trains a lightweight LLaMA-based language model
  - Generates melody from scratch with top-k sampling


## 🔹 2. Conditioned Generation (Melody-to-Harmony via BiLSTM)

A PyTorch-based deep learning model that generates harmony conditioned on melody input.

- **Task**: Predict appropriate chord sequences given a monophonic melody.
- **Technique**: Bidirectional LSTM tagger with embeddings and positional awareness
- **Dataset**: Bach Chorales (via `music21`)
- **Output**: Chord sequences aligned to melody, exportable to MIDI.
- **Highlights**:
  - Sequence labeling using BiLSTM
  - Train/validation split with padding and collate functions
  - Custom dataset and training loop
  - Evaluation based on chord accuracy


---

## 🧰 Tech Stack

- **Languages**: Python 3.8+
- **Deep Learning**: PyTorch, MiniLLaMA (transformer-based model)
- **Music Processing**: Music21, REMI-style tokenization
- **Data Formats**: MIDI (symbolic), ABC notation
- **Other**: NumPy, tqdm, collections


---


## 🧪 Setup

Install dependencies:

```bash
pip install torch numpy music21 tqdm

