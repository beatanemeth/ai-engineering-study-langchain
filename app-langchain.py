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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from utils_logging import log_info, log_warn, log_error, log_step


# -----------------------------------------------------
# ⚙️ Configurations
# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_FILE_PATH = os.path.join(current_dir, "data", "brain_facts_book.pdf")
CHROMA_STORAGE_PATH = os.path.join(current_dir, "chroma_persistent_storage")
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


# Helper: format retrieved documents into a single context string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 1. RAG Answer Prompt
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert neuroscience assistant.  
Use ONLY the following retrieved context to answer the user question.

If the answer is not fully supported by the context, say:
"I cannot find this information in the provided book context."

Retrieved Context:
{context}

User Question:
{question}

Final Answer:
"""
)

# 2. Define the main Generation Logic (answer generation only)
# This sub-chain prepares the context and generates the answer string.
rag_answer_chain = (
    # A. Pass input ('question') and retrieve context. The context is formatted by 'format_docs'.
    RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    # B. Apply the Answer Prompt (receives {context} and {question})
    | ANSWER_PROMPT
    # C. LLM generates answer
    | llm_model
    # D. Parse into string
    | StrOutputParser()
)

# 3. Build the final `agent_executor` (The main RAG chain that returns sources)
# Use RunnableParallel (LCEL syntax) to execute two chains simultaneously and return a dictionary.
agent_executor = RunnableParallel(
    # A. 'output' key: The final string answer from the LLM
    output=rag_answer_chain,
    # B. 'source_documents' key: The raw list of documents from the retriever
    #    (Ensures the retriever runs with the same 'question' input)
    source_documents=itemgetter("question") | retriever,
)

# 4. Define the executable `chain` (Input Mapping)
# Maps the external 'input' key (from CLI) to the internal 'question' key.
chain = {"question": itemgetter("input")} | agent_executor


# -----------------------------------------------------
# ⚙️ QUERY EXECUTION
# -----------------------------------------------------
def execute_query(chain, query):  # Renamed 'agent_executor' to 'chain' for clarity
    log_step("--- ⚙️ Query Execution ---")
    print(f"QUESTION: {query}")

    # .invoke() runs the entire pipeline, returning the dictionary from RunnableParallel
    # The input must be a dictionary matching the chain's expected input (i.e., 'input')
    result = chain.invoke({"input": query})

    # Access the keys defined in the RunnableParallel: 'output' and 'source_documents'
    answer = result["output"]
    sources = result["source_documents"]

    print(f"ANSWER: {answer}")

    # Print the source documents
    log_step(f"Sources Used ({len(sources)} Chunks):")
    for i, doc in enumerate(sources):
        page_num = doc.metadata.get("page", "N/A")
        print(f"Source_{i+1} (Page: {page_num}):")
        print(f"  {doc.page_content[:200]}...")
        print("-" * 20)


# -----------------------------------------------------
# CLI Support
# -----------------------------------------------------
# Pass any new question as a command-line argument to test
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        execute_query(chain, user_query)
    else:
        log_warn("No question provided as argument.")
        log_info("Example:")
        print('   python3 your-rag-app.py "What is the corpus callosum?"')
