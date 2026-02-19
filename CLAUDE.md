# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses **uv** for Python environment management. All commands should use `uv` command or use the uv-managed Python interpreter (`.venv/Scripts/python.exe` on Windows) .

## Project Overview

This is CS336 Spring 2024 Assignment 1: Basics - a deep learning/NLP assignment focused on implementing foundational components of a Transformer language model from scratch. The project requires implementing various neural network components and matching them against reference implementations.

## Common Commands

```bash
# 使用uv运行文件
uv run main.py

# 使用 uv 运行测试
uv run pytest

# 使用 uv 安装依赖
uv add -r requirements.txt
uv add -r requirements-test.txt

# 运行特定测试文件
uv run pytest tests/test_tokenizer.py

# 运行特定测试
uv run pytest tests/test_tokenizer.py::test_roundtrip_empty -v
```

## Architecture

The project has two main parts:

1. **Core implementations** (`cs336_basics/`):
   - `tokenizer.py` - BPE tokenizer (GPT-style), with encode/decode methods
   - `bpe.py` - BPE training algorithm with `train_bpe()` function

2. **Test adapters** (`tests/adapters.py`):
   - Contains stub functions that students must implement to connect their code to tests
   - Key functions to implement: `run_positionwise_feedforward`, `run_scaled_dot_product_attention`, `run_multihead_self_attention`, `run_transformer_lm`, `run_rmsnorm`, `run_gelu`, `run_cross_entropy`, `run_adamw`, `run_get_lr_cosine_schedule`, `run_gradient_clipping`, `run_get_batch`, `get_tokenizer`

Tests verify implementations against reference implementations (PyTorch, tiktoken). The architecture uses "adapter" functions that load reference weights and compare outputs with student implementations.

## Data

Training data should be downloaded to `data/` directory:
- TinyStories dataset (from HuggingFace)
- OpenWebText sample (from HuggingFace)
