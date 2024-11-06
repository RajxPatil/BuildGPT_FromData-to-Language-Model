# Building a Large Language Model from Scratch

This repository contains code, datasets, and notebooks for building, training, and experimenting with Language Models (LMs). It specifically focuses on constructing a GPT-style architecture and implementing tokenization techniques. The project is ideal for those who want hands-on experience creating foundational LLM components and experimenting with custom datasets.

## Project Overview

### What Is This Project?

This project is an educational initiative to build essential components of large language models from scratch. It explores tokenization techniques like Byte-Pair Encoding (BPE) and dives into the structure and training of a simple generative pre-trained transformer (GPT) model. The project helps demystify the inner workings of LLMs by implementing core features step-by-step.

### How It Works

1. **Tokenization**: Uses BPE and bigram tokenization techniques to convert raw text data into token sequences that a neural network can process.
2. **Model Training**: This implements a simple GPT-like model and trains it on a sample dataset (`wizard_of_oz.txt`). The training scripts (training.py and related notebooks) provide options for experimenting with model configurations, adjusting hyperparameters, and fine-tuning.
3. **Chatbot Interface**: A chatbot (`chatbot.py`) allows interaction with the trained model. It demonstrates how LLMs generate text based on input prompts.
4. **Data Extraction Scripts**: Scripts like `data-extract-v2.py` and `data-extract-v3.py` show how to preprocess and structure data for LLM training.
5. **Sample Notebooks**: These contain example code to visualize model outputs, experiment with tokenization techniques, and test other NLP tasks.

### Problem It Solves

The project serves as a hands-on learning tool for understanding and building LLMs. By implementing tokenization, model training, and interactive applications, users can gain insights into how LLMs process text, learn language structures and generate coherent responses. This understanding is essential for researchers and developers working on NLP tasks or custom LLMs.

# Tech Stack

Here’s a brief overview of the technologies used in this project:

## Programming Language
- Python

## Deep Learning Framework
- PyTorch

## Tokenization Techniques
- Byte-Pair Encoding (BPE)
- Bigram

## Libraries
- NumPy
- PyTorch
- pylzma
- ipykernel
- Jupyter

## Model
- GPT-style Architecture

## Development Tools
- Jupyter Notebooks
- Visual Studio 2022

## Environment
- CUDA (optional)
- CPU fallback

## Version Control
- Git

## Folder Structure

- `bigram.ipynb`: Notebook for exploring bigram tokenization.
- `bpe-v1.ipynb`: Notebook demonstrating Byte-Pair Encoding for tokenization.
- `chatbot.py`: Python script to interact with the trained model in a chatbot format.
- `data-extract-v2.py` & `data-extract-v3.py`: Scripts for processing raw text data into structured datasets.
- `gpt-v1.ipynb` & `gpt-v2.ipynb`: Notebooks implementing and training a simple GPT model.
- `torch-examples.ipynb`: Notebook with PyTorch examples for understanding tensors, layers, and basic operations.
- `training.py`: Script to train the GPT model on the sample dataset.
- `bpe_tokenizer.json`: JSON file defining the vocabulary for BPE tokenization.
- `vocab.txt`: Vocabulary file for tokenization purposes.
- `wizard_of_oz.txt`: Sample dataset containing text for model training.

## Setup and Installation

1. **Dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Additional Dependencies**:
   ```bash
   pip install pylzma numpy ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118
3. **Environment**
   Set device parameter to use cuda if a GPU is available, or default to cpu for non-GPU setups.
   Install Visual Studio 2022 (for LZMA compression): [Download Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)

## Running the Project
1. **Tokenization**:
    Run bpe-v1.ipynb or bigram.ipynb to tokenize the text data.
2. **Model Training**:
    Use training.py or the gpt-v1.ipynb notebook to train a basic GPT model on the tokenized data.
3. **Chatbot Interaction**:
   Run chatbot.py to interact with the trained model. The chatbot generates responses based on your input, showcasing the model’s text generation capabilities.

## Resources and Further Reading
1. Attention is All You Need: [Read the Paper](https://arxiv.org/pdf/1706.03762.pdf)
2. Survey of LLMs: [Read the Survey](https://arxiv.org/pdf/2303.18223.pdf)
3. QLoRA for Efficient LLM Finetuning: [Read QLoRA paper](https://arxiv.org/pdf/2305.14314.pdf)



