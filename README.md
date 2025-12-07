# 🦜 LangChain RAG Project: Simple QA vs. Continuous Chat

> A project that advances previous RAG work - [AI Engineering - Study RAG](https://github.com/beatanemeth/ai-engineering-study-rag) - by utilizing **LangChain** to orchestrate RAG pipelines, including a demonstration of how to implement **Conversational Memory** for contextual, multi-turn chat.

## Table of Contents

1.  [Project Overview & Learning Goals](#1-project-overview--learning-goals-)
2.  [Architectures Implemented](#2-architectures-implemented-🧠)
3.  [Technical Stack](#3-technical-stack-🛠️)
4.  [Prerequisites](#4-prerequisites-📦)
5.  [Getting Started](#5-getting-started-🚀)
    - [5.1. Configuration (.env)](#51-configuration-env)
    - [5.2. Download Knowledge Base](#52-download-knowledge-base)
    - [5.3. Setup Python Environment](#53-setup-python-virtual-environment)
    - [5.4. Install Dependencies](#54-install-dependencies)
    - [5.5. Run the Applications](#55-run-the-applications)
6.  [LangChain RAG Flow Details](#6-langchain-rag-flow-details-⚙️)

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

<br></br>

## 4. Prerequisites 📦

You must have the following installed and configured:

- **Python 3.x**
- An **OpenRouter API Key**

<br></br>

## 5. Getting Started 🚀

### 5.1. Configuration (`.env`)

1.  Create a file named `.env` in the project's root directory.
2.  Populate the file with your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-Your_OpenRouter_API_Key
```

⚠️ **Security Tip**: Never commit your `.env` file to version control.

### 5.2. Download Knowledge Base

1. Download a copy of [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) PDF.
2. Name the file exactly as: `brain_facts_book.pdf`
3. Place it inside the project's `/data `folder.

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

### 5.4. Install Dependencies

With the virtual environment active, install all necessary packages from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

### 5.5. Run the Application

You can now run either the naive or the HyDE implementation. **The indexing phase will only run the first time** and populate the `chroma_persistent_storage` folder.

```Bash
# Run the Naive RAG implementation
python3 app-rag-naive.py

#OR

# Run the HyDE RAG implementation
python app-rag-hyde.py
```

When you are finished, exit the isolated environment:

```Bash
deactivate
```

<br></br>

## 6. LangChain RAG Flow Details ⚙️

### Indexing (RAG Step 1)

The PDF is loaded using `PyPDFLoader`, chunked using `RecursiveCharacterTextSplitter`, and then the chunks are embedded by the local **Sentence Transformer model** and persisted in **ChromaDB**.

### Simple QA Chain (`app-langchain.py`)

This uses the powerful LCEL:

1. The user `input` is mapped to the `question` key.

2. `RunnableParallel` runs two paths:

   - **Generation Path**: `question` -> `retriever` -> `format_docs` -> `context` is used in `ANSWER_PROMPT` -> `llm_model` -> `output` (final answer).

   - **Source Path**: `question` -> `retriever` -> `source_documents` (raw chunks).

### Conversational RAG Chain (`app-langchain-chat.py`)

This chain introduces memory to handle context-dependent questions:

1. **History-Aware Retrieval**: The user's new `input` and the `chat_history` are passed to a prompt (`CONTEXTUALIZE_Q_PROMPT`). The LLM uses this to rewrite the input into a standalone query (e.g., "_What about the hippocampus?_" becomes "What is the primary function of the hippocampus?").

2. **Retrieval**: The standalone query is passed to the `retriever` to find relevant chunks.

3. **Generation**: The context, `chat_history`, and original input are combined in the `FINAL_ANSWER_PROMPT`, and the LLM generates the response.

4. **Memory Management**: `RunnableWithMessageHistory` intercepts the input/output and automatically updates the `Chat Message History` object for the next turn.
