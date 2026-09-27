# Handbook RAG

A small command-line retrieval-augmented generation (RAG) demo. It reads a
handbook, splits it into overlapping text chunks, embeds those chunks, and
retrieves the chunks most relevant to a question.

The current program is a **retriever**, not a complete generative chatbot: it
prints matching handbook text. `RAGPipeline` can optionally call an LLM through
an `llm_call` function, but no LLM provider is configured in this project.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- Internet access the first time the embedding model is run, so
  `sentence-transformers` can download it

The embedding model used by default is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
It converts text and questions into vectors so the program can rank handbook
chunks by similarity. The model is downloaded and cached automatically by
Sentence Transformers.

## Setup

From the project directory, create and activate a virtual environment in
PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

If PowerShell blocks activation, you can skip activation and use the virtual
environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

## Add or update the handbook

Place your handbook text in `company_handbook.txt` next to `main.py`. The
program also accepts the original filename `company_handbook` without an
extension. Replace the sample content with information you want the program
to search.

For useful answers, include clear, factual sections such as vacation and sick
leave policies, working hours, holidays, expenses, benefits, and contact
details. The system can only retrieve information present in the handbook.

## Run

```powershell
python main.py
```

Or, if you did not activate the virtual environment:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Type a question at the prompt, for example:

```text
How many paid vacation days do employees receive?
```

The program prints the most relevant retrieved passage(s). To get a
natural-language generated response instead, pass an LLM provider function to
`RAGPipeline`; this project does not currently include an LLM provider or API
configuration.

## How it works

1. `chunk_text` divides the handbook into 200-word chunks with a 50-word
   overlap.
2. `Embedder` converts each chunk and the question into normalized vectors
   using `all-MiniLM-L6-v2`.
3. `VectorStore` ranks chunks by vector similarity and returns the top three.
4. `RAGPipeline` returns those passages, or passes them to an optional LLM
   callback.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Chunking, embedding, vector search, and command-line entry point |
| `company_handbook.txt` | Source text to search |
| `requirements.txt` | Python package dependencies |