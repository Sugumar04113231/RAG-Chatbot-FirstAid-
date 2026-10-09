# app.py
import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from pathlib import Path
from transformers import pipeline

# ---------------- CONFIG ---------------- #
INDEX_DIR = Path("index_data")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"

# Using TinyLlama for better RAG performance on CPU
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOP_K = 3

# ---------------- STARTUP ---------------- #
app = FastAPI()

# Global variables for models and data
state = {}

@app.on_event("startup")
def startup():
    if not FAISS_PATH.exists() or not META_PATH.exists():
        print("ERROR: Index files not found. Run build_index.py first.")
        return

    print("Loading FAISS index and metadata...")
    state["index"] = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "r") as f:
        state["metadata"] = json.load(f)

    print("Loading embedding model...")
    state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loading LLM ({LLM_MODEL})...")
    state["generator"] = pipeline(
        "text-generation",
        model=LLM_MODEL,
        device=-1 # Use CPU
    )

class QueryIn(BaseModel):
    query: str

@app.post("/query")
async def handle_query(payload: QueryIn):
    if "index" not in state:
        raise HTTPException(status_code=500, detail="Index not loaded.")

    query = payload.query
    
    # 1. Retrieve
    query_vec = state["embed_model"].encode([query])
    distances, indices = state["index"].search(np.array(query_vec).astype('float32'), TOP_K)
    
    context_chunks = [state["metadata"][i] for i in indices[0] if i != -1]
    context_text = "\n\n".join(context_chunks)

    # 2. Augment & Generate
    # Note: Using the TinyLlama Chat template for better results
    prompt = f"""<|system|>
You are a helpful First Aid Assistant. Use the provided context to answer the user's question accurately. 
If the information is not in the context, tell the user to seek professional medical help immediately.
Context:
{context_text}</s>
<|user|>
{query}</s>
<|assistant|>"""

    response = state["generator"](
        prompt, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    full_text = response[0]['generated_text']
    # Extract only the assistant's part
    answer = full_text.split("<|assistant|>")[-1].strip()

    return {"answer": answer, "sources": context_chunks}

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>First Aid AI Assistant</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        .header { background: #d32f2f; color: white; padding: 1rem; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 15px; line-height: 1.5; font-size: 15px; }
        .user { align-self: flex-end; background: #007bff; color: white; border-bottom-right-radius: 2px; }
        .bot { align-self: flex-start; background: white; color: #333; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .input-area { background: white; padding: 20px; display: flex; gap: 10px; border-top: 1px solid #ddd; }
        input { flex: 1; padding: 12px; border: 1px solid #ccc; border-radius: 25px; outline: none; }
        button { background: #d32f2f; color: white; border: none; padding: 10px 25px; border-radius: 25px; cursor: pointer; font-weight: bold; }
        button:hover { background: #b71c1c; }
        .loading { font-style: italic; color: #666; font-size: 13px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header"><h1>First Aid RAG Assistant</h1></div>
    <div id="chat-container">
        <div class="message bot">Hello! I am your First Aid assistant. How can I help you today? (e.g., "What to do for a burn?")</div>
    </div>
    <div class="input-area">
        <input type="text" id="q" placeholder="Type your medical emergency question here..." onkeypress="if(event.key==='Enter') ask()">
        <button onclick="ask()">Send</button>
    </div>

<script>
async function ask() {
    const input = document.getElementById('q');
    const query = input.value.trim();
    if (!query) return;

    const container = document.getElementById('chat-container');
    
    // User Message
    const userDiv = document.createElement('div');
    userDiv.className = 'message user';
    userDiv.textContent = query;
    container.appendChild(userDiv);
    input.value = '';
    
    // Bot Loading
    const botDiv = document.createElement('div');
    botDiv.className = 'message bot';
    botDiv.innerHTML = '<span class="loading">Thinking...</span>';
    container.appendChild(botDiv);
    container.scrollTop = container.scrollHeight;

    try {
        const res = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        const data = await res.json();
        botDiv.textContent = data.answer;
    } catch (e) {
        botDiv.textContent = "Error: Could not connect to the server.";
    }
    container.scrollTop = container.scrollHeight;
}
</script>
</body>
</html>
"""
