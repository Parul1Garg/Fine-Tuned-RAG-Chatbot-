import json, requests


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""
You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

If the answer is not in the context, respond professionally:
"The document does not contain this information. Please try rephrasing your query."
Answer:
"""


def generate_with_ollama(prompt, model="mistral:7b"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
