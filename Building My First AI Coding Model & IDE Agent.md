# Building My First AI Coding Model & IDE Agent

## Phase 1 -- Project Kickoff

### Goal

Build a custom AI coding assistant similar to Cursor, Composer, Codex,
or Antigravity by fine-tuning an open-source coding model and
integrating it into a custom IDE agent.

# Step 1: Define the Project Scope

Before writing code or training a model, decide exactly what the
assistant should do.

### Initial Version (MVP)

The first version should be able to:

-   Answer coding questions
-   Generate code
-   Fix bugs
-   Explain code
-   Generate FastAPI endpoints
-   Generate React Native components
-   Suggest improvements

### Do NOT Build Yet

Avoid these advanced features initially:

-   Autonomous coding agents
-   Multi-agent systems
-   Terminal execution
-   Git automation
-   Long-term memory
-   Complex planning systems

Focus on building a strong coding assistant first.

# Step 2: Setup Development Environment

### Required Software

Install:

#### Python

Version:

    Python 3.11+

Check:

    python --version

#### Git

Check:

    git --version

#### VS Code

Install latest VS Code.

Useful extensions:

-   Python
-   Pylance
-   Jupyter
-   GitLens

# Step 3: Create Project Structure

Create a project folder:

    mkdir my-coding-model
    cd my-coding-model

Create:

    my-coding-model/
    │
    ├── datasets/
    ├── models/
    ├── notebooks/
    ├── training/
    ├── inference/
    ├── evaluation/
    ├── docs/
    └── requirements.txt

# Step 4: Create Python Environment

### Mac/Linux

    python -m venv venv
    source venv/bin/activate

### Windows

    python -m venv venv
    venv\Scripts\activate

# Step 5: Install Core Libraries

Install:

    pip install torch
    pip install transformers
    pip install datasets
    pip install accelerate
    pip install peft
    pip install trl
    pip install jupyter

Verify:

    python -c "import torch; print(torch.__version__)"

# Step 6: Learn Core Concepts

Before training anything, understand:

### Topics

1.  Tokens
2.  Embeddings
3.  Transformers
4.  Attention Mechanism
5.  Fine-Tuning
6.  LoRA
7.  QLoRA
8.  RAG
9.  Inference
10. Quantization

### Outcome

You should be able to explain:

-   What a token is
-   What a transformer does
-   Difference between training and inference
-   Why QLoRA saves GPU memory

# Step 7: Select Base Model

Recommended first model:

### Option A (Recommended)

Qwen Coder 7B

Why:

-   Excellent coding performance
-   Easier to train
-   Lower GPU requirements

### Option B

DeepSeek Coder 6.7B

### Option C

Gemma Coding Models

# Step 8: Download and Test Base Model

Before training:

-   Download model
-   Run locally
-   Ask coding questions

Example:

    Create a FastAPI CRUD API for Products.

Observe:

-   Quality
-   Speed
-   Memory usage

This becomes your baseline.

# Step 9: Create Initial Dataset

Create folder:

    datasets/

Create file:

    coding_dataset.jsonl

Example record:

    {
      "instruction": "Create FastAPI CRUD API",
      "input": "Product Model",
      "output": "Complete FastAPI CRUD implementation"
    }

### Dataset Categories

Collect examples for:

-   FastAPI
-   React Native
-   PostgreSQL
-   Authentication
-   JWT
-   Docker
-   API Integration
-   Debugging
-   Refactoring
-   Code Review

Goal:

100 examples initially.

# Deliverables for Phase 1

By the end of Step 1 you should have:

✅ Development environment ready

✅ Python environment created

✅ Core libraries installed

✅ Base model selected

✅ Base model running locally

✅ Dataset folder created

✅ First 100 training examples collected

# Success Criteria

Do not move to training until:

1.  Model runs locally.
2.  Dataset structure is finalized.
3.  At least 100 high-quality examples exist.
4.  You understand LoRA and QLoRA concepts.

After completing this phase, the next phase will be: "Fine-Tuning Qwen
Coder with QLoRA and Evaluating Results."
