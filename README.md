# Neural Text Degeneration — Nucleus Sampling

Reproduction and analysis of **"The Curious Case of Neural Text Degeneration"** (Holtzman et al., ICLR 2020), which introduces **Nucleus Sampling** (top-p) — now the standard decoding strategy in virtually all large language models.

## Overview

This project implements Nucleus Sampling from scratch in PyTorch and runs two small-scale experiments on GPT-2 medium to explore the paper's central claims.

**The core problem:** maximization-based decoding (beam search, greedy) produces repetitive, degenerate text, while pure sampling produces incoherent gibberish. The paper argues this is not a model flaw but an intrinsic property of the decoding strategy.

**The solution:** instead of sampling from a fixed top-k tokens or the full distribution, Nucleus Sampling dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold p — the *nucleus*. This adapts the candidate pool to the model's confidence at each step.

## Experiments

**Experiment 1 — Repetition rate across decoding methods**
Measures the fraction of generations that fall into repetition loops, across greedy, temperature, top-k, and top-p sampling. Qualitatively reproduces Figure 9 of the paper: greedy and low-temperature decoding produce the most degeneration, while nucleus sampling in its standard range (p ∈ [0.9, 0.95]) stays near 0%.

**Experiment 2 — Dynamic nucleus size**
Tracks the nucleus size |V^(p)| at each generation step. The nucleus ranges from 1 to over 5,000 tokens within a single generation, spanning 4 orders of magnitude. This directly visualizes why a fixed k in top-k sampling is fundamentally inadequate — no single value can cover both the high-confidence and high-uncertainty regimes.

## Reference

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020. https://arxiv.org/abs/1904.09751
