# Developer Guide

Welcome to the **Semantic Hybrid Agent**. This guide explains the architecture, design decisions, and code flow for developers who want to understand, extend, or debug the system.

## 1. Architecture Overview

The agent is designed as an **Autonomous System** powered by **Semantic Kernel**. Unlike traditional RAG systems that use rigid "Routers" (if/else statements), this agent uses **LLM Function Calling** to dynamically decide how to solve a problem.

### The "Big Three" Components
1.  **The Brain (`src/core/agent.py`)**:
    - Uses **`FunctionChoiceBehavior.Auto()`**: This is the universal standard in Semantic Kernel v1.x+. It allows the kernel to translate the Agent's "tool needs" into the specific API dialect of the backend LLM (whether it's OpenAI functions or Gemini tools).
    - **Grounding**: Dynamically scans `data/inputs/` and injects the list of available files into the System Prompt. This gives the agent "self-awareness" of its resources.
    - **Safety**: The System Prompt explicitly forbids the agent from attempting to read files directly (e.g. `open('secret.txt')`). It forces the use of the `AdvancedRagPlugin` (which only reads indexed chunks) and `DataAnalystPlugin` (which only reads pre-loaded CSVs).
    - It sees a list of tools (Plugins) and decides which ones to call.
    - It can call multiple tools in a meaningful sequence (e.g., "Get Data" -> "Get Context" -> "Synthesize").

2.  **The Engine (`src/core/kernel.py`)**:
    - Configures the Semantic Kernel.
    - Connects to the LLM (Ollama or Gemini).
    - Connects to the Vector Service (ChromaDB).
    - **Key Decision**: We use `KnowledgeRecord` (`src/core/models.py`) to strictly define what "memory" looks like in Chroma (Content + Vector + Metadata).

3.  **The Filters (`src/core/ranker.py`)**:
    - **Problem**: Vector search is fast but "fuzzy". It often retrieves irrelevant docs.
    - **Solution**: We use a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to essentially "grade" the top 25 vector results and pick only the top 5 high-quality matches. This drastically reduces hallucinations.

---

## 2. Plugin System

We organize capabilities into **Plugins**. This is idiomatic Semantic Kernel design.

### A. AdvancedRagPlugin (`src/plugins/rag_plugin.py`)
- **Role**: The Librarian.
- **Function**: `SearchKnowledge(query: str)`
- **Logic**:
    1.  Embeds the query.
    2.  Searches ChromaDB (Top 25).
    3.  **Re-ranks** the results using `RankerService`.
    4.  Returns the top 5 citations.
- **Why Re-rank?**: To ensure the "Context" fed to the LLM is extremely relevant, preventing "garbage in, garbage out".

### B. DataAnalystPlugin (`src/plugins/data_plugin.py`)
- **Role**: The Analyst.
- **Function**: `AnalyzeData(query: str)`
- **Logic**:
    1.  Loads all `.csv` files from `data/inputs/` into Pandas DataFrames.
    2.  Uses the `csv_analyst` prompt (`config/prompts.yaml`) to write Python code.
    3.  **Executes** the code in a sandbox (`exec()`).
    4.  Returns the raw result.
- **Key Feature**: The **Synthesis Loop**. The Agent doesn't just dump the number. It takes the number + the user's question and "synthesizes" a natural language sentence.

---

## 3. Data Flow: "The Life of a Query"

Scenario: *"What is the revenue in the CSV and does it cover the budget defined in the PDF?"*

1.  **Input**: User sends the query to `api.py`.
2.  **Orchestration (`agent.py`)**:
    - The LLM receives the message.
    - It sees two tools: `AnalyzeData` and `SearchKnowledge`.
3.  **Execution (Auto-Pilot)**:
    - **Thought 1**: "I need the revenue." -> Calls `AnalyzeData("calculate total revenue")`. -> Result: `$5,000`.
    - **Thought 2**: "I need the budget." -> Calls `SearchKnowledge("project budget policy")`. -> Result: `"The budget cap is $4,000"`.
4.  **Synthesis**:
    - The LLM combines these findings.
    - **Response**: "The revenue is $5,000, which exceeds the budget cap of $4,000 mentioned in the policy."

---

## 4. Key Design Decisions

### Why Separate Pipelines?
We treat PDFs and CSVs differently:
- **PDFs** are **Unstructured**. They belong in a Vector DB.
- **CSVs** are **Structured**. Vector search is terrible at "What is the average?". Math requires **Code Execution**.
- **Result**: We get the best of both worlds.

### Why Semantic Kernel?
It provides a robust abstraction for **Connectors** (easy switch between OpenAI/Ollama/Gemini) and **Function Calling** (the backbone of our auto-pilot).

### Why ChromaDB?
- It's local (no cloud bills).
- It's fast.
- It integrates natively with Semantic Kernel.

---

## 5. Directory Guide

- **`config/`**: Control center.
  - `prompts.yaml`: **First-Class Asset**. This is the Agent's "System Instructions". It defines:
    - **Identity**: Who the agent is (e.g., "Economic Expert").
    - **Boundaries**: What it should REFUSE to do.
    - **Defaults**: How to handle vague queries (e.g., "Summarize" -> "Search the PDF").
  - `settings.yaml`: Hardware/Service config (LLM Providers).
- **`data/`**: The "Brain's hippocampus". Inputs go here. DB lives here.
- **`src/core/`**: Infrastructure (Agent, Kernel, DB Schema).
- **`src/plugins/`**: Capabilities (RAG, Data).
- **`src/utils/`**: Helpers.

---

## 6. How to Extend

- **Add a Tool**: Create `src/plugins/my_new_plugin.py`, define a `@kernel_function`, and register it in `agent.py`.
- **Change the LLM**: Just update `config/settings.yaml`. You can now assign different models to different roles using the `roles` config:
    ```yaml
    roles:
      agent: "gemini" # Smart Orchestrator
      tools: "ollama" # Fast Utility Worker
    ```
- **Improve RAG**: Tune `rerank_top_k` in `config/settings.yaml`.
