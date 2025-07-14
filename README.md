# Distributed Training Practice
Hands-on PyTorch examples to learn and demonstrate (multi-)GPU model training and inference.

## 1. Distributed Data Parallel (DDP) Training with PyTorch
This notebook provides a concise and practical walkthrough of implementing Distributed Data Parallel (DDP) training using PyTorch. DDP is an efficient way to scale model training across multiple GPUs, offering near-linear speedup and better performance compared to DataParallel.

## 2. Mini-GPT with multi-GPU training
This project demonstrates how to train a character-level GPT model on the Tiny Shakespeare dataset using PyTorch with support for multi-GPU distributed training via DistributedDataParallel (DDP). It is designed as a practical, modular framework for experimenting with distributed deep learning techniques and efficient training setups.

### Technical Architecture ###

<img width="1257" height="668" alt="tech arch" src="https://github.com/user-attachments/assets/4717d923-eba6-43ca-8e14-cedc3d3f1c84" />

## 3. SGLang Demo
This code runs Qwen2-VL-7B-Instruct, a vision-language model (VLM), using the SGLang framework on Modal, a serverless GPU cloud platform. It allows users to ask natural language questions about images via an HTTP API, and the model returns text-based answers by understanding both the image and the question.

                           ┌────────────────────────────┐
                           │     User / Client (API)    │
                           │----------------------------│
                           │ Sends POST request:        │
                           │ {                          │
                           │   "image_url": "...",      │
                           │   "question": "What is..." │
                           │ }                          │
                           └────────────┬───────────────┘
                                        │
                                        ▼
                           ┌────────────────────────────┐
                           │   Modal FastAPI Endpoint   │
                           │  (Model.generate method)   │
                           └────────────┬───────────────┘
                                        │
           Downloads image from URL     │
           ────────────────────────────►│
                                        │
                                        ▼
                      ┌────────────────────────────────────┐
                      │     SGLang Runtime (Qwen2-VL)      │
                      │------------------------------------│
                      │  1. Load model & tokenizer         │
                      │  2. Format prompt using template   │
                      │  3. Attach image + user question   │
                      │  4. Generate assistant response    │
                      └────────────────────┬───────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────┐
                      │  Response: Textual answer to     │
                      │  the visual question             │
                      └──────────────────────────────────┘
                                           │
                                           ▼
                           ┌────────────────────────────┐
                           │    User / Client receives  │
                           │    JSON answer string      │
                           └────────────────────────────┘




