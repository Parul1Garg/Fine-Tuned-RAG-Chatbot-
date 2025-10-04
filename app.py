import streamlit as st
from src.retriever import load_corpus, retrieve
from src.generator import build_prompt, generate_with_ollama  


st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("Local RAG Chatbot with Ollama")

if "corpus_loaded" not in st.session_state:
    try:
        st.session_state.chunks, st.session_state.embs, st.session_state.index, st.session_state.embedder = load_corpus()
        st.session_state.corpus_loaded = True
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        st.info("Make sure you've run `python ingest.py` first")
        st.stop()

MODEL_NAME = "mistral:7b"

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask something about the document...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.history.append(("user", query))
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            retrieved_chunks = retrieve(
                query, 
                st.session_state.chunks, 
                st.session_state.index, 
                st.session_state.embedder, 
                top_k=3
            )
            
            prompt = build_prompt(query, retrieved_chunks)
            
            for token in generate_with_ollama(prompt, model=MODEL_NAME):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
            response_placeholder.markdown(error_msg)
            full_response = error_msg
    
    st.session_state.history.append(("assistant", full_response))