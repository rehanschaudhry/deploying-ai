# DSI RAG Assistant

## What This Is
The objective of this project is to build a Retrieval-Augmented Generation (RAG) assistant that can ingest various types of documents, chunk them effectively, and use a vector database to enable efficient retrieval for question-answering tasks. The system will be designed to handle different file formats, preserve metadata.

## Current Status
Deploying AI was the course I was taking from DSI, and I wanted to showcase what I learned in the course by building a RAG assistant. I have so far built a parser for Jupyter notebooks, PDFs, Python files, and markdown files to extract text content and relevant metadata.

## Architecture
The architecture of the RAG assistant consists of several key components:
1. **Document Ingestion**: This component is responsible for parsing various document formats and extracting the content and metadata. Within the documents, I am extracting the text content and relevant metadata from juypter notebooks, PDFs, python files and markdown files. I have implemented a one-parser-per-file-type strategy to ensure that the unique characteristics of each file format are handled appropriately. This allows for more accurate extraction and better preservation of metadata, which is crucial for effective retrieval later on.
2. **Chunking**: This component breaks down the extracted content into smaller, manageable pieces (chunks) that can be stored in the vector database. The chunking strategy is designed to balance the size of the chunks with the amount of context preserved, which is important for ensuring that retrieved information is relevant and useful for answering user queries.
3. **Vector Database**: The chunked content will be stored in ChromaDB, which allows for efficient retrieval based on vector similarity. Each chunk is associated with its metadata to enable context-aware retrieval. 
4. **Query Interface**: This component allows users to input questions and retrieves relevant chunks from ChromaDB to generate answers. The retrieval process takes into account the metadata to ensure that the most relevant information is returned.
5. **MLflow Integration**: MLflow is used to track experiments, monitor performance metrics, and manage the lifecycle of the RAG assistant. This allows for continuous improvement and optimization of the system based on empirical data.

## Tech Stack
- **Programming Language**: Python 
- **Vector Database**: ChromaDB 
- **Experiment Tracking**: MLflow
- **Document Parsing Libraries**: 
    PDF: pyMuPDF for parsing and extracting content and metadata from PDF files. 
    Jupyter notebooks: nbformat for parsing and extracting content and metadata. 
    Python files and markdown files: Reads a single file as plain UTF-8 text and saves a JSON record with metadata. Whole file is one unit; the chunker in Step 6 will handle splitting.

## Project Roadmap

The full project plan is 14 steps. Here's where I am:

### Phase 1: Document Ingestion (Steps 1-5) — DONE
- [x] Step 1: Environment setup, Claude API + MLflow verified
- [x] Step 2: File walker producing inventory
- [x] Step 3: Notebook parser
- [x] Step 4: PDF parser + inspect utility
- [x] Step 5: Code and markdown parser

### Phase 2: Chunking and Storage (Steps 6-9) — IN PROGRESS
- [ ] Step 6: Chunker with metadata
- [ ] Step 7: First MLflow experiment (chunk size 512)
- [ ] Step 8: ChromaDB setup
- [ ] Step 9: Embeddings for all parsed files

### Phase 3: Retrieval and Answer Generation (Steps 10-14)
- [ ] Step 10: Retriever with top-k similarity search
- [ ] Step 11: Answer generator with citation prompt
- [ ] Step 12: End-to-end test with 5 real questions
- [ ] Step 13: Chunk size experiments (256 vs 512 vs 1024)
- [ ] Step 14: MLflow registry for v1.0

## Project Structure
rag_assistant/
├── scripts/
│   ├── verify_setup.py
│   ├── walk_repo.py
│   ├── parse_notebook.py       ← Step 3
│   ├── parse_pdf.py            ← Step 4
│   ├── inspect_pdf.py          ← Step 4 utility
│   └── parse_text_file.py      ← Step 5
├── parsed_notebooks/
│   └── 01_1_introduction_parsed.json    (1 file)
├── parsed_pdfs/
│   └── 05_rag_parsed.json               (1 file)
├── parsed_python/
│   └── 05_src__course_chat__app_py_parsed.json   (1 file)
└── parsed_markdown/
    └── README_md_parsed.json            (1 file)

## How to Run
1. Ensure you have Python installed (version 3.11 or higher).
2. Set up a virtual environment (optional but recommended): `python -m venv venv` and activate it.
3. Fork and clone the repository: `git clone https://github.com/rehanschaudhry/deploying-ai`
4. Install dependencies by running: `pip install -r requirements.txt`
5. Start MLflow: in a second terminal with the venv active, run mlflow ui --port 5000. Open http://localhost:5000 in your browser.
6. Run the document ingestion scripts to parse the files and extract content and metadata:
   - `deploying-ai/rag_assistant/scripts/parse_notebook.py` for Jupyter notebooks
   - `deploying-ai/rag_assistant/scripts/parse_pdf.py` for PDFs
   - `deploying-ai/rag_assistant/scripts/inspect_pdf.py` to inspect the parsed PDF content and metadata
   - `deploying-ai/rag_assistant/scripts/parse_text_file.py` for Python and markdown files
7. Verify that the parsed content and metadata are correctly stored in the respective directories (`parsed_notebooks/`, `parsed_pdfs/`, `parsed_python/`, `parsed_markdown/`).

## Key Decisions so Far
- **Choice of Vector Database**: The reason why I chose ChromaDB was it was taughted in the course and wanted to showcase what I learned in the course. ChromaDB is a powerful vector database that allows for efficient storage and retrieval of high-dimensional data, which is essential for the RAG assistant to function effectively. It provides robust support for similarity search, which is crucial for retrieving relevant chunks of information based on user queries.
- **MLflow Integration**: Even though we did not learn MLflow in the class, MLflow is wildly popular experimental tracking tool that was developed by Databricks. I just does not track experiments but also allows you to manage the lifecycle of your machine learning models, which is crucial for a project like this where we are continuously improving the RAG assistant based on user interactions and performance metrics. Recently they have also track LLM experiments which is relevant to this project. [x]
- **One-Parser-Per-File-Type Strategy**: I decided to implement a separate parser for each file type to ensure that the unique characteristics of each format are handled appropriately, leading to more accurate extraction and better preservation of metadata.

## What I Learned

### About my own learning
I learned that I'm a dialogue learner — pointed questions help me absorb material better than reading documentation cover-to-cover. One thing that I learned is that there is a lot of pre-processing that goes into making a RAG system work, and I had underestimated the importance of document parsing and metadata extraction in the overall architecture. I did not know that parsing different file types required unique approaches. 

### About code as an artifact
The code for parsing PDFs was more complex than I expected, especially when it came to preserving metadata and handling different layouts. I assumed that reading a PDF vs a Jupyter notebook would be similar, but I quickly realized that the structure of PDFs can vary widely, and extracting meaningful content while preserving context is a non-trivial task. 
Setting up code locally versus in GitHub Codespaces revealed differences in how environments handle dependencies and configuration. Codespaces required more careful setup than expected, especially around secrets management and the setup.sh initialization script.I encountered unexpected issues related to file encoding and BOM characters, which affected how configuration files were read in the environment. I also learned that dependency management can differ across platforms, requiring adjustments to requirements.txt to ensure compatibility within the Codespaces Linux environment.

### About Python implementation
I am still developing fluency in Python and have been building experience through work and independent projects over the past year. I am comfortable with core constructs such as functions, classes, data structures, and libraries, but writing production-ready code from scratch remains a challenge.

Early in the project, I relied on LLMs to generate complete solutions, which often resulted in code I could not fully explain or debug efficiently when issues arose. I shifted to a more structured workflow during the RAG assistant project by decomposing problems into smaller components aligned with the system design and requesting targeted code generation per step. I validate each output by reviewing, understanding, and testing it incrementally before integration.

I also learned that selecting appropriate libraries is critical for correctness and maintainability. For example, I used PyMuPDF for PDF parsing due to its reliable support for extracting structured content and metadata.

This approach reinforced that robust Python development depends on understanding implementation details, especially when handling file formats and edge cases.

### About tooling and reproducibility  
I initially assumed a single requirements.txt file was sufficient to guarantee consistent environments across local development and GitHub Codespaces. This assumption broke down when dependency installation behaved differently across environments.

During setup in Codespaces, I encountered failures caused by version mismatches and platform-specific dependencies that were not apparent locally. This highlighted that dependency declarations alone do not ensure reproducibility.

The experience made clear the value of using a lock file (e.g., requirements-lock.txt) to pin exact dependency versions and reduce variability introduced by transitive dependencies or environment differences.

Adopting a locked dependency strategy would improve reproducibility across development, CI, and cloud environments such as Codespaces, and reduce time spent debugging environment-related inconsistencies.

### Things that surprised me
The things that surprised me the most is what I predicted how many characters would be produced after parsing the PDFs. I expected to get a lot more text content out of the PDF , but I was surprised to find that the actual amount of extracted text was much less than I anticipated. This was due to the fact that both PDFs can contain a lot of non-text elements (like images, code cells, and formatting) that do not contribute to the text content. Additionally, the structure of these documents can make it challenging to extract meaningful text without losing important context or metadata. This experience highlighted the importance of having a robust parsing strategy that can effectively handle different document formats while preserving relevant information for downstream tasks like chunking and retrieval. 