# Fine-Tuned-RAG-Chatbot- 

##  Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that can answer questions about a provided PDF document ( *AI Training Document*).  
It combines **semantic search with embeddings** and a **local LLM (via Ollama)** to generate context-aware answers with **streaming responses** in a Streamlit app.

---

##  Project Architecture & Flow

The chatbot follows a modular **RAG (Retrieval-Augmented Generation)** architecture:

### 1. Document Ingestion (`src/ingest.py`)
- Load raw PDF (`/data` folder) using **PyPDF**.
- Clean and normalize text.
- Split into ~200-word chunks (with overlap of size 30).
- Generate semantic embeddings with **all-MiniLM-L6-v2**.
- Store:
  - Chunks → `/chunks/chunks.json`
  - Embeddings → `/vectordb/embeddings.npy`
  - Vector index (FAISS) → `/vectordb/faiss.index`

### 2. Retriever (`src/retriever.py`)
- Loads chunks, embeddings, and FAISS index.
- Encodes user query with the same embedding model.
- Performs semantic similarity search to retrieve top-k=3 relevant chunks.

### 3. Generator (`src/generator.py`)
- Constructs a **prompt template**:
- Sends the prompt to a local **LLM mistral:7b via Ollama**.
- Streams model output token-by-token for real-time responses.

### 4. RAG Pipeline and Streamlit App (`app.py`)
- Orchestrates retrieval + generation.
- Provides an easy `.ask(query)` method to get grounded responses.
- Web interface for interactive chat.
- Features:
- Chat-style UI (`st.chat_input`, `st.chat_message`).
- Streams responses in real time.
- Displays user history across turns.
- Error handling if Ollama is not running.
 ----
 
##  End-to-End Flow
1. Run `ingest.py` → Preprocess documents, build embeddings & FAISS index.
2. Run `app.py` → Launch Streamlit chatbot.
3. User asks a question:
   - Retriever finds relevant chunks.
   - Generator builds prompt with context.
   - LLM (via Ollama) generates a streamed answer.
4. Answer is displayed live in the chat interface, grounded in the source document.

-----

##  Model & Embedding Choices
- **Embedding Model**:  
  [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) → Fast, lightweight, and effective for semantic similarity search.  
- **Vector Database**:  
  [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search on embeddings.  
- **LLM for Answer Generation**:  
  [Mistral-7B](https://mistral.ai/) via [Ollama](https://ollama.ai/) for local inference.

-----

## Instructions to run the Chatbot with streaming response enabled

### 1. Create & Activate environment 
- In Anaconda prompt, write - conda activate AI
  
### 2. Install dependencies
- pip install numpy pypdf faiss-cpu sentence-transformers streamlit

### 3. Run ingest.py file on Spyder
- cd C:\Users\HP\OneDrive\PARUL\AI_Chatbot\src
- python ingest.py

### 4. Start Ollama (Local LLM)
- Install [Ollama](https://ollama.ai/) from Google Chrome.  
- Pull the Mistral model on Windows Powershell as: ollama pull mistral:7b

### 5. Launch the streamlit app on Anaconda prompt
- conda activate AI
- cd C:\Users\HP\OneDrive\PARUL\AI_Chatbot\
- streamlit run app.py

### 6. Chat with Streaming Responses
- Enter a query in the chat input box.
- The system retrieves relevant chunks, builds a context-aware prompt, and streams the LLM’s answer token by token.
- The answer will appear live, word by word, similar to ChatGPT.

----

## Sample queries and output screenshots 

![OUTPUT4](https://github.com/user-attachments/assets/85b6bd0c-c423-4eb6-b7ef-afba4d3a25ce)

![OUTPUT3](https://github.com/user-attachments/assets/ef7c7d2b-1976-486d-b5f2-44b73a2b3ceb)

![OUTPUT2](https://github.com/user-attachments/assets/6ccca0e3-65f9-424b-a811-db10d00fce25)

<img width="1314" height="601" alt="OUTPUT1" src="https://github.com/user-attachments/assets/69379c1d-e340-4356-a86a-6351f5cb956f" />

-----

## Demo Video

[Watch the full demo](https://github.com/Parul1Garg/Fine-Tuned-RAG-Chatbot-/blob/main/Screenshots/Screencast%20from%2004-10-25%2004_45_00%20PM%20IST.webm)

