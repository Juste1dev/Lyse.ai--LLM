# Training Lyse.ai

This document explains how to train Lyse.ai and how to contribute training runs to the project.

Lyse.ai currently uses a two-stage training pipeline:

1. Pre-training
2. Supervised Fine-Tuning (SFT)

The current model is Lyse-67M.

---

# Requirements

Recommended environment:

* Python 3.10 or newer
* NVIDIA GPU with CUDA support
* CUDA-compatible PyTorch
* At least 16 GB of system RAM
* Enough disk space for datasets and checkpoints

More GPU VRAM generally allows larger batches and can improve training performance.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Juste1dev/Lyse.ai--LLM.git
cd Lyse.ai--LLM
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Verify that PyTorch can detect your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# Training pipeline

The general Lyse training pipeline is:

```text
Dataset
   ↓
Tokenizer
   ↓
Pre-tokenization
   ↓
Pre-training
   ↓
Pre-trained checkpoint
   ↓
SFT dataset
   ↓
Supervised Fine-Tuning
   ↓
SFT checkpoint
```

---

# 1. Build the tokenizer

Build the tokenizer using:

```bash
python scripts/build_tokenizer.py
```

The tokenizer is used by the training pipeline and SFT process.

---

# 2. Pre-tokenize the corpus

Prepare the pre-training corpus:

```bash
python scripts/pretokenize_corpus.py
```

Make sure the dataset and configuration files are correctly configured before starting training.

---

# 3. Pre-training

Start pre-training with:

```bash
python scripts/train.py --config configs/base_67m.yaml
```

The configuration file controls the model and training parameters.

The default configuration is:

```text
configs/base_67m.yaml
```

Additional configurations can be stored in the `configs/` directory.

---

# 4. Resume pre-training

If training is interrupted, you can automatically resume from the latest checkpoint:

```bash
python scripts/train.py --config configs/base_67m.yaml --resume auto
```

You can also specify a checkpoint manually:

```bash
python scripts/train.py --config configs/base_67m.yaml --resume path/to/checkpoint.pt
```

---

# 5. Build the SFT dataset

After pre-training, prepare the supervised fine-tuning dataset:

```bash
python scripts/build_sft_dataset.py
```

The SFT dataset is used to train the model to follow instructions and generate conversational responses.

---

# 6. Supervised Fine-Tuning

Start SFT from a pre-trained checkpoint:

```bash
python scripts/train_sft.py --config configs/base_67m.yaml --base-checkpoint path/to/pretrained_checkpoint.pt
```

The `--base-checkpoint` argument specifies the pre-trained model that will be used as the starting point for SFT.

---

# 7. Resume SFT

To automatically resume an interrupted SFT run:

```bash
python scripts/train_sft.py --config configs/base_67m.yaml --base-checkpoint path/to/pretrained_checkpoint.pt --resume auto
```

---

# Training contributions

Contributors with available GPU resources are welcome to train Lyse and submit useful training results.

Before starting a large training run, make sure you are using the latest version of the repository and the currently recommended configuration.

For every training contribution, provide:

```text
GPU:
GPU VRAM:
CPU:
RAM:
Python version:
PyTorch version:
CUDA version:

Training stage:
Configuration:
Dataset:
Number of steps:

Final training loss:
Final validation loss:

Checkpoint:
```

Additional information such as training duration and throughput is also useful.

---

# Experimental training

Experimental configurations are welcome.

If you change an important training parameter, clearly document the modification.

Examples include:

* Learning rate
* Batch size
* Gradient accumulation
* Sequence length
* Optimizer
* Dataset mixture
* Number of steps
* Weight decay
* Model architecture

Experimental results should contain enough information for another contributor to reproduce the experiment.

---

# Checkpoints

Do not commit large checkpoints directly to GitHub.

Large model files should be hosted using the project's designated model hosting platform.

When submitting a training contribution, provide the checkpoint location and the corresponding training information.

---

# Reproducibility

Reproducibility is important for Lyse.ai.

When possible, contributors should provide:

* Exact configuration
* Dataset version or source
* Training steps
* Hardware
* Software versions
* Training and validation metrics
* Checkpoint

This allows different training runs to be compared reliably.

---

# Training issues

If training fails, open a GitHub Issue and include:

* Command used
* Configuration
* Error message
* GPU
* PyTorch version
* CUDA version
* Relevant logs

Do not remove error information from logs before submitting an issue.

---

# Contributing training results

To contribute a training run:

1. Train Lyse using the documented process.
2. Record the training configuration and results.
3. Keep the resulting checkpoint.
4. Open an Issue or Pull Request describing the run.
5. Provide the checkpoint location.
6. Wait for the maintainers to review the results.

Training results may be evaluated before being used as official Lyse checkpoints.

---

# Current training scripts

The main training scripts are:

```text
scripts/build_tokenizer.py
scripts/pretokenize_corpus.py
scripts/build_sft_dataset.py
scripts/train.py
scripts/train_sft.py
```

The main configuration files are located in:

```text
configs/
```

For questions or proposed changes to the training pipeline, open a GitHub Issue or Discussion.
