import os
from dotenv import load_dotenv
import sys

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils_logging import log_info, log_warn, log_error, log_step


# -----------------------------------------------------
# ⚙️ Configurations
# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_FILE_PATH = os.path.join(current_dir, "data", "brain_facts_book.pdf")
CHROMA_STORAGE_PATH = os.path.join(current_dir, "chroma_persistent_storage_chat")
COLLECTION_NAME = "neuroscience_rag_collection"
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Load environment variables from .env file
load_dotenv()

# The 'openai' library is used for OpenRouter,
# but we use the OpenRouter API Key and base URL.
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    log_error("Missing OPENROUTER_API_KEY in environment.")


# -----------------------------------------------------
# 🧩 LangChain Component Initialization
# Defining the core components: LLM, embeddings model and vector store
# -----------------------------------------------------
log_step("--- 🧩 Initializing LangChain Components ---")
## 1. LLM via OpenRouter
# OpenRouter implements the OpenAI API standard, so we use the `ChatOpenAI` class (which is compatible with the OpenAI API standard).
# It automatically reads the OPENROUTER_API_KEY from the environment.
log_info(f"Initializing LLM via OpenRouter: {OPENROUTER_MODEL}")
llm_model = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

## 2. Embeddings Model
# This instantiates the local, free Sentence Transformer model.
# The `model_kwargs={'device': 'cpu'}` is a good practice for local setups
# to ensure it uses the CPU if you don't have a configured GPU.
log_info("Initializing SentenceTransformer, a local embeddings model: all-MiniLM-L6-v2")
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"}
)

## 3. Vector Store (Chroma)
# Conditionally initialize the vectorstore object by loading it if it exists.
if os.path.exists(CHROMA_STORAGE_PATH):
    log_info(
        f"Vector Store already exists at {CHROMA_STORAGE_PATH}. Loading persistent store."
    )
    # Initialize the vectorstore variable by loading the persisted store
    vectorstore = Chroma(
        persist_directory=CHROMA_STORAGE_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    # Set a flag to skip indexing
    chroma_store_exists = True
else:
    log_info("No persisted Chroma DB found. Will be created in Indexing Phase.")
    # Initialize the vectorstore variable to None (or just let the Indexing Phase define it)
    vectorstore = None
    # Set a flag to trigger indexing
    chroma_store_exists = False

# -----------------------------------------------------
# 📄 RAG STEP 1 — INDEXING PHASE
# Load → Chunk → Embed (Vectorize) → Store
# -----------------------------------------------------

if not os.path.exists(PDF_FILE_PATH):
    log_error(f"PDF not found: {PDF_FILE_PATH}")
    sys.exit(1)

# Check the flag set during initialization
if not chroma_store_exists:

    log_step("--- 📄 Indexing Phase (RAG Step-1) ---")

    ## 1. Load: Load the PDF
    log_info(f"1. Loading document from {PDF_FILE_PATH}...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    log_info(f"Loaded {len(documents)} pages.")

    ## 2. Chunk: Split the document
    log_info("2. Splitting documents into chunks...")
    # `RecursiveCharacterTextSplitter` is often the best default choice
    # as it tries to split on paragraphs, then sentences, etc., preserving context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=100,  # Overlap helps preserve context between chunks
    )

    # Read the text content of each file and store it with metadata. --> Here we chunk the pages of the single PDF and add metadata
    chunked_docs = []

    for doc in documents:
        # Split the page into chunks
        chunks = text_splitter.split_documents([doc])

        for chunk in chunks:
            # Add metadata to each chunk indicating its source
            chunk.metadata["source_file"] = os.path.basename(PDF_FILE_PATH)
            chunk.metadata["page"] = doc.metadata.get(
                "page", doc.metadata.get("page_number")
            )

            chunked_docs.append(chunk)

    log_info(f"Created {len(chunked_docs)} chunks for indexing.")
    if chunked_docs:
        print("Example chunk:", chunked_docs[0].page_content[:200], "...")

    ## 3. Embed & 4. Store: Create the Chroma Vector Store
    log_info(
        f"3. & 4. Creating and persisting Chroma Vector Store at {CHROMA_STORAGE_PATH}..."
    )
    # Chroma handles the embedding of the chunks and their storage in one step
    # Note: This is where 'vectorstore' is defined if it was 'None' earlier
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=CHROMA_STORAGE_PATH,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()

# -----------------------------------------------------
# 🔍 RAG STEP 2 — RETRIEVAL PHASE
# Retrieve Chunks
# -----------------------------------------------------
log_step("--- 🔍 Retrieval Phase (RAG Step-2) ---")

# Define the threshold for acceptable similarity score (e.g., 0.7)
# This model's scores are typically in the range [0, 1].
SIMILARITY_THRESHOLD = 0.1  # You may need to tune this value
TOP_K_CHUNKS = 4  # Retrieve up to 4 highly-scoring chunks for context

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": TOP_K_CHUNKS,
        "score_threshold": SIMILARITY_THRESHOLD,
    },
)

# -----------------------------------------------------
# 💡 RAG STEP 3 — GENERATION PHASE
# -----------------------------------------------------
log_step("--- 💡 Generation Phase (RAG Step-3) ---")

# 1. Contextualizing Question Prompt (for Query Rewriting)
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, generate a standalone question that can be used to search the vector store. Do not answer the question, just rephrase it if necessary. If no history exists, return the question as is.",
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Placeholder for the memory buffer
        ("human", "{input}"),
    ]
)

# 2. History-Aware Retriever Chain
# This chain takes the user input and history, generates a standalone query, and uses it for retrieval.
history_aware_retriever = create_history_aware_retriever(
    llm_model, retriever, CONTEXTUALIZE_Q_PROMPT
)

# 3. Final Answer Prompt (must include history)
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert neuroscience assistant. Use ONLY the following retrieved context and the chat history to answer the user question. If the answer is not fully supported by the context, say: 'I cannot find this information in the provided book context.'\n\nRetrieved Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 4. Define the Document Combination Chain
# This chain is responsible for stuffing the documents into the prompt and calling the LLM.
# It uses the LLM and the final prompt.
combine_docs_chain = create_stuff_documents_chain(
    llm_model,
    FINAL_ANSWER_PROMPT,  # This prompt must contain {context} and {chat_history}
)

# 5. Final Conversational Retrieval Chain
# This chain combines the history-aware retrieval and the document combination chain.
# It now correctly takes 2 positional arguments:
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever,  # 1st argument (the retriever)
    combine_docs_chain,  # 2nd argument (the generation chain)
)

# 6. Define the final runnable with message history (Same as before)
chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    lambda session_id: ChatMessageHistory(session_id=session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# -----------------------------------------------------
# ⚙️ QUERY EXECUTION
# -----------------------------------------------------
# Define a single, constant session ID for all queries in a single run
SESSION_ID = "neuroscience_session"


def execute_query(chain, query, session_id):
    log_step("--- ⚙️ Query Execution ---")
    print(f"QUESTION: {query}")

    # .invoke() requires a 'session_id' in the config dictionary
    result = chain.invoke(
        {"input": query}, config={"configurable": {"session_id": session_id}}
    )

    # Access the keys defined by create_retrieval_chain
    answer = result["answer"]
    # The documents are returned in the 'context' key by create_retrieval_chain
    sources = result["context"]

    print(f"ANSWER: {answer}")

    # Print the source documents (same logic as before)
    log_step(f"Sources Used ({len(sources)} Chunks):")
    for i, doc in enumerate(sources):
        page_num = doc.metadata.get("page", "N/A")
        print(f"Source_{i+1} (Page: {page_num}):")
        print(f"  {doc.page_content[:200]}...")
        print("-" * 20)


# -----------------------------------------------------
# CLI Support (Continuous Chat & Termination)
# -----------------------------------------------------
# Define a single, constant session ID for all queries in a single run
SESSION_ID = "neuroscience_session"


def execute_query(chain, query, session_id):
    log_step("--- ⚙️ Query Execution ---")
    print(f"QUESTION: {query}")

    # .invoke() requires a 'session_id' in the config dictionary
    result = chain.invoke(
        {"input": query}, config={"configurable": {"session_id": session_id}}
    )

    # Access the keys defined by create_retrieval_chain
    answer = result["answer"]
    # The documents are returned in the 'context' key by create_retrieval_chain
    sources = result["context"]

    print(f"ANSWER: {answer}")

    # Print the source documents
    log_step(f"Sources Used ({len(sources)} Chunks):")
    for i, doc in enumerate(sources):
        page_num = doc.metadata.get("page", "N/A")
        print(f"Source_{i+1} (Page: {page_num}):")
        print(f"  {doc.page_content[:200]}...")
        print("-" * 20)


if __name__ == "__main__":

    # 1. Handle single query from command line arguments
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        print(f"\n--- Running Single Command-Line Query ---")
        execute_query(chain, user_query, SESSION_ID)
        # We don't break here; the interactive loop starts right after.

    # 2. Enter the interactive chat loop
    print("\n--- 🧠 Neuroscience Chat Initiated ---")
    print(f"Session ID: {SESSION_ID}")
    print("Ask a question about the brain.")
    print("Type **'quit'**, **'exit'**, or **'no'** to end the chat.")
    print("-" * 40)

    while True:
        try:
            # Get user input
            user_input = input("USER > ")

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "no"]:
                print("--- Chat Session Ended. Goodbye! 👋 ---")
                break

            if not user_input.strip():
                continue

            # Execute the query, maintaining the same SESSION_ID for memory
            execute_query(chain, user_input, SESSION_ID)

        except KeyboardInterrupt:
            print("\n--- Chat Session Ended by User (Ctrl+C). Goodbye! 👋 ---")
            break
        except Exception as e:
            log_error(f"An unexpected error occurred: {e}")
            break
