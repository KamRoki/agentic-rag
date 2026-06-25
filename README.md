# Agentic RAG
Agentic RAG is an experimental Retrieval-Augmented Generation system built with Python, LangChain and LangGraph. It demonstrates an agentic RAG workflow that can route questions between a local vector store and web search, evaluate retrieved documents, generate answers and check whether the final response is grounded in the available context.

## What this project does
This repository implements a graph-based RAG workflow with the following capabilities:
* routes user questions either to a local vector store or to web search
* retrieves documents from a local Chroma collection
* grades retrieved documents for relevance
* enriches weak or incomplete retrieval results with Tavily web search
* generates concise answers from the retrieved context
* checks whether the generated answer is grounded in the documents
* checks whether the answer resolves the original user question
* exports the LangGraph workflow visualization to `graph.png`.

## Architecture
The main workflow is defined in `graph/graph.py` and compiled as a LangGraph application.

```text
agentic-rag/
├── graph/
│   ├── chains/
│   │   ├── answer_grader.py
│   │   ├── generation.py
│   │   ├── hallucination_grader.py
│   │   ├── retrieval_grader.py
│   │   └── router.py
│   ├── nodes/
│   │   ├── generate.py
│   │   ├── grade_documents.py
│   │   ├── retrieve.py
│   │   └── web_search.py
│   ├── consts.py
│   ├── graph.py
│   └── state.py
├── ingestion.py
├── main.py
├── graph.png
├── pyproject.toml
├── uv.lock
└── README.md
```

## Workflow
The application follows this decision flow:
1. A user question is passed into the graph.
2. The router decides whether the question should use:
   - the local vector store, or
   - Tavily web search.
3. If the local vector store is selected, documents are retrieved from Chroma.
4. Retrieved documents are graded for relevance.
5. If the retrieved documents are not relevant enough, the system adds web search results.
6. The generator creates an answer from the available context.
7. The generated answer is checked for grounding against the documents.
8. The answer is checked against the original question.
9. If the answer is useful, the workflow ends.
10. If the answer is not useful or not grounded, the graph either retries generation or uses web search.

## Tech stack
* Python 3.11
* LangChain
* LangGraph
* Chroma
* OpenAI models and embeddings
* Tavily Search
* python-dotenv
* uv

## Requirements
This project requires Python `>=3.11,<3.12`.
You also need API keys for OpenAI and Tavily.
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Installation
Clone the repository:

```bash
git clone https://github.com/KamRoki/agentic-rag.git
cd agentic-rag
```

Install dependencies with `uv`:

```bash
uv sync
```

## Preparing the vector store
The ingestion script loads source documents from selected web pages, splits them into chunks and prepares them for storage in Chroma.
The current `ingestion.py` file contains the Chroma creation block commented out. To build the local vector store, uncomment this section:

```python
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma-data",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)
```

Then run:

```bash
uv run python ingestion.py
```

This creates a local `.chroma` directory with the persisted vector store.
After the vector store has been created, the retriever can be used through:

```python
retriever = Chroma(
    collection_name="rag-chroma-data",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
```

## Running the app
Run the main script:

```bash
uv run python main.py
```

The current example question is defined in `main.py`:

```python
{
    "question": "Example user question?"
}
```

The application prints the graph output in the terminal.

## Generating the workflow graph
To export the LangGraph workflow visualization, run:

```bash
uv run python graph/graph.py
```

This generates or updates:

```text
graph.png
```

## Main components

### `ingestion.py`
Loads web documents, splits them into chunks and connects the project to a local Chroma vector store.

### `graph/chains/router.py`
Routes the user question to either the vector store or web search. The vector store is intended for questions related to AI, RAG, agents, prompt engineering and adversarial attacks.

### `graph/nodes/retrieve.py`
Retrieves relevant documents from the local Chroma collection.

### `graph/nodes/grade_documents.py`
Grades retrieved documents for relevance. If a document is not relevant, the graph marks the state for web search.

### `graph/nodes/web_search.py`
Runs Tavily Search and adds the search results to the document context.

### `graph/nodes/generate.py`
Generates the final answer from the question and available context.

### `graph/chains/hallucination_grader.py`
Checks whether the generated answer is supported by the retrieved documents.

### `graph/chains/answer_grader.py`
Checks whether the generated answer addresses the original user question.

## Example usage

```python
from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

result = app.invoke(
    input={
        "question": "What is LLM Wiki concept?"
    }
)

print(result)
```

## Author
Kamil Stachurski (e-mail: kamil.stachurski@icloud.com)
