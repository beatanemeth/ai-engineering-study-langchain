# 🦜 LangChain RAG Project: Simple QA vs. Continuous Chat

> A project that advances previous RAG work - [AI Engineering - Study RAG](https://github.com/beatanemeth/ai-engineering-study-rag) - by utilizing **LangChain** to orchestrate RAG pipelines, including a demonstration of how to implement **Conversational Memory** for contextual, multi-turn chat.

## Table of Contents

1.  [Project Overview & Learning Goals](#1-project-overview--learning-goals-)
2.  [Architectures Implemented](#2-architectures-implemented-🧠)
3.  [Technical Stack](#3-technical-stack-🛠️)
4.  [Prerequisites](#4-prerequisites-📦)
5.  [Getting Started](#5-getting-started-🚀)
6.  [Resources](#6-resources-)

<br></br>

## 1. Project Overview & Learning Goals 🎯

This project transitions from plain Python code to using the **LangChain framework** to streamline the RAG process, focusing on flexibility and advanced features like memory.

**Key Learning Objectives:**

- **LangChain Expression Language (LCEL):** Implementing RAG chains using the standard `|` (pipe) operator with components like `RunnablePassthrough` and `RunnableParallel`.
- **Conversational RAG:** Understanding how to integrate chat history into the RAG cycle to handle follow-up questions (e.g., "What about its primary function?").
- **History-Aware Retrieval:** Using LangChain primitives (`create_history_aware_retriever`) to rewrite search queries based on prior context.
- **LLM/Embedding/Vector Store Integration:** Configuring and coordinating core RAG components through LangChain wrappers.

<br></br>

## 2. Architectures Implemented 🧠

This repository provides two distinct, executable RAG chains built with LangChain:

| File                    | Technique              | Use Case                                           | Key LangChain Components                                                          |
| :---------------------- | :--------------------- | :------------------------------------------------- | :-------------------------------------------------------------------------------- |
| `app-langchain.py`      | **Simple RAG Chain**   | Single Question-Answer (QA) via CLI argument.      | **LCEL, `RunnableParallel`**, `ChatPromptTemplate`                                |
| `app-langchain-chat.py` | **Conversational RAG** | Multi-turn, stateful chat session in the terminal. | `create_retrieval_chain`, `MessagesPlaceholder`, **`RunnableWithMessageHistory`** |

<br></br>

## 3. Technical Stack 🛠️

| Component           | Detail                                                                        | Use                                                           |
| :------------------ | :---------------------------------------------------------------------------- | :------------------------------------------------------------ |
| **Orchestration**   | **LangChain**                                                                 | Manages the sequence of RAG steps and component interactions. |
| **LLM Provider**    | OpenRouter: `google/gemma-3-27b-it:free`                                      | Generates the final answer and handles query rewriting.       |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2`                                      | Runs locally to create vector representations.                |
| **Vector DB**       | [Chroma](https://docs.trychroma.com/) (Persistent)                            | Stores and indexes the document embeddings.                   |
| **PDF Loader**      | `PyPDFLoader` (LangChain Wrapper)                                             | Loads and extracts text content from the PDF.                 |
| **Source Data**     | [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) (PDF) | The sole knowledge base for grounding answers.                |
| **Development OS**  | Linux Mint 21.2                                                               | The system used for development.                              |

<br></br>

## 4. Prerequisites 📦

You must have the following installed and configured:

- **Python 3.10.12+**
  > ⚠️ **Version Note:** This project was developed and tested using **Python 3.10.12**. While most dependencies will work with newer versions (e.g., Python 3.11/3.12), it is recommended using Python 3.10 or a compatible version to ensure environmental stability.
- An **OpenRouter API Key** (Set as `OPENROUTER_API_KEY` in the `.env` file).

<br></br>

## 5. Getting Started 🚀

### 5.1. Download Knowledge Base

1. Download a copy of [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) PDF.
2. Name the file exactly as: `brain_facts_book.pdf`
3. Replace the empty `brain_facts_book.pdf` file inside the project's `/data `folder with your downloaded sample.

### 5.2. Configuration (`.env`)

1.  In the root directory of this project, rename the `.env.example` to `.env`.
2.  Populate the file with your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-Your_OpenRouter_API_Key
```

⚠️ **Security Tip**: Never commit your `.env` file to version control.

### 5.3. Setup Python Virtual Environment

It is best practice to use a virtual environment to isolate project dependencies.

#### 1. Create the environment

Run the following command in your project directory:

```Bash
python3 -m venv .venv
```

#### 2. Activate the environment

macOS/Linux:

```Bash
source .venv/bin/activate
```

Windows (Command Prompt):

```Bash
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```Bash
.venv\Scripts\Activate.ps1
```

Your command prompt will now show the environment name, like `(.venv) user@host:~/project$`, indicating that it is active.

### 5.4. Update pip

```Bash
python -m pip install --upgrade pip
```

### 5.5. Install Dependencies

With the virtual environment active, install all necessary packages from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

### 5.6. Run the Application

You can now run either the naive or the HyDE implementation.

```Bash
# Single question
python3 app-langchain.py "Your question here."

#OR

# Continuous chat
python3 app-langchain-chat.py
```

### 5.7. Deactivate the environment

When you are finished, exit the isolated environment:

```bash
deactivate
```

Your command prompt will return to its default state, and the environment name `(.venv)` will disappear.

<br></br>

## 6. Resources 📚

[Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)

[Get started with Chroma vector stor](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)

[PyPDFLoader](https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader)

[ChromaDB](https://docs.trychroma.com/docs/overview/introduction)

[Sentence Transformer](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

[OpenRouterAi](https://openrouter.ai/)
