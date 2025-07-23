RAGBot: Private RAG Chatbot with Web Search
Overview
RAGBot is a privacy-focused Retrieval-Augmented Generation (RAG) chatbot that answers user queries using a local document set and falls back to web search when necessary. It uses a locally-run Gemma model (gemma3:1b-it-qat) via Ollama, a Chroma vectorstore for document retrieval, and the Tavily API for web searches. The Streamlit interface provides a conversational UI with document upload, query input, log viewing, and response metrics.
Architecture

Document Ingestion (app/rag_engine/ingest.py): Loads TXT, PDF, and DOCX files from data/, splits them into chunks, and builds a Chroma vectorstore using HuggingFace embeddings (sentence-transformers/all-mpnet-base-v2).
Query Processing (app/rag_engine/query.py): Uses LangChain to retrieve relevant document chunks, query the Gemma model, or perform web searches via tavily_tool. Includes citation formatting and fallback logic.
Logging (app/utils/logger.py): Logs queries, answers, sources, response times, and citations to eval_logs.json.
Web Search (app/websearch/tavily_tool.py): Handles web searches when local documents are insufficient (assumed functional).
Streamlit UI (main.py): Provides a chat interface (user messages left, bot responses right), document upload, log display, and a "Metrics" button for debug details.

Features

Conversational chat with history.
File upload for TXT, PDF, DOCX to update the vectorstore.
Displays web links or document citations in responses.
"Metrics" button to show source, response time, retrieved chunks, and context.
Log viewer for evaluation metrics.

Installation

Clone the Repository:
git clone <repository-url>
cd RAGBot


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Ensure requirements.txt includes:
streamlit
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-ollama
python-dotenv
pypdf
python-docx

Install with:
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the root directory with your Tavily API key:
TAVILY_API_KEY=your_api_key


Install Ollama:

Follow instructions at Ollama to install and run the gemma3:1b-it-qat model locally.
Ensure Ollama is running before starting the app.



Usage

Prepare Documents:

Place TXT, PDF, or DOCX files in the data/ directory (sample files: ceo_list.txt, elon_musk.txt, openai_info.txt).
Alternatively, upload files via the Streamlit sidebar.


Run the Streamlit App:
streamlit run main.py


Access the app at http://localhost:8501.


Interact with the Chatbot:

Use the sidebar to upload documents, which are processed into the vectorstore/.
Enter queries in the chat input box.
View responses with citations (document names or web URLs).
Click "Metrics" to see debug details (source, response time, chunks, context).
Check logs in the sidebar for evaluation metrics.



Directory Structure
RAGBot/
├── app/
│   ├── data/
│   │   └── chroma/
│   │       └── chroma.sqlite3
│   ├── rag_engine/
│   │   ├── ingest.py
│   │   └── query.py
│   ├── utils/
│   │   ├── eval_logs.json
│   │   └── logger.py
│   └── websearch/
│       └── tavily_tool.py
├── data/
│   ├── ceo_list.txt
│   ├── elon_musk.txt
│   ├── openai_info.txt
│   └── sample.txt.bak
├── vectorstore/
├── .env
├── .gitignore
├── main.py
├── README.md
├── requirements.txt

Notes

Ensure the Tavily API key is valid for web searches.
The Gemma model must be running via Ollama.
Logs are stored in app/utils/eval_logs.json and can be viewed in the Streamlit sidebar.
The app maintains privacy by running the LLM locally and only using external APIs (Tavily) when necessary.
