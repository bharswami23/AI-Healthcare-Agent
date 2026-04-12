import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import requests
import streamlit as st
import time
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import torch
torch.set_default_tensor_type(torch.FloatTensor)

@st.cache_resource
def load_models():
    models = {}

    models["all_mini_lm"] = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cpu"
    )
    """
    models["bge_base"] = SentenceTransformer(
        "BAAI/bge-base-en-v1.5", device="cpu"
    )

    models["intfloat"] = SentenceTransformer(
        "intfloat/e5-base-v2", device="cpu"
    )

    models["bge_large"] = SentenceTransformer(
        "BAAI/bge-large-en-v1.5", device="cpu"
    )
    """
    return models

models = load_models()

global API_KEY, model, qa_list, documents, embeddings, conversation_history,index
llmmodel=""
def safe_llm_call(response):
    try:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", None)
        if content is None:
            print("⚠️ Empty content from model:", data)
            return ""
        return content.strip()
    except Exception as e:
        print("⚠️ Exception parsing response:", e)
        return ""

def call_with_retry(payload, retries=3):
    for i in range(retries):
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
        }, json=payload)
        resp = safe_llm_call(response)
        if resp:
            return resp
        print(f"Retry {i+1}...")
    return ""

def safe_llm_call(response):
    try:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", None)

        if content is None:
            return None

        return content.strip()

    except Exception as e:
        print("⚠️ Parsing error:", e)
        return None
def call_with_retry(payload, retries=3):
    for i in range(retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )

            content = safe_llm_call(response)

            if content:   # ✅ valid response
                return content

            print(f"⚠️ Empty response, retry {i+1}")

        except Exception as e:
            print(f"⚠️ Request failed (retry {i+1}):", e)

    return ""   # final fallback
def query_decomp_tool(query: str, model=llmmodel):
    prompt = f"""
    You are an expert query/task planner for a RAG system.

    Given the user query/task:
    {query}

    Decompose it into high-quality subquestions optimized for document retrieval.

    Requirements:
    - Include key entities, technical terms, and keywords explicitly
    - Avoid pronouns or references to other subquestions
    - Ensure semantic diversity (no overlap or redundancy)
    - Include subquestions that retrieve supporting statistics, percentages, and empirical evidence RELEVANT to the query.
    - The subquestions should help answer the main question comprehensively when combined.
    - Use retrieval-friendly phrasing (who, what, when, which, how many, etc.)
    - Expand queries with synonyms or alternate formulations if useful
    - Cover all aspects of the original query (exhaustive)
    - MAXIMUM 8 subquestions, MINIMUM 3. This requirement is to be strictly followed.

    ADDITIONAL REQUIREMENTS (CRITICAL):
    - If the query involves reasoning, explanation, or causal inference:
        - Include subquestions that capture relationships (cause, effect, comparison)

    STRICT OUTPUT FORMAT (MANDATORY):
    <subquestions>
    - ...
    - ...
    - ...
    </subquestions>

    Include reasoning and explanation outside of the <subquestions> tags in the format:
    <reasoning>
    - ....
    - ....
    - ....
    </reasoning>
    Include no other text or formatting outside of the specified tags.
    """
    prompt_ = f"""You are an expert query planner for a RAG system.
    Given a user question {query}, help answer it by decomposing it into minimal, non-overlapping and exhaustive subquestions optimized for document retrieval.
    The subquestions should be enclosed within <subquestions></subquestions> tags and formatted as a bulleted list with the bullet '-'.
    Just print the subquestion in each bulleted item with no other text within the tags <subquestions></subquestions>.
    For each subquestion: make it atomic, include key entities and keywords explicitly in the subquestions, avoid pronouns or references 
    to other subquestions in each subquestion, ensure semantic diversity to maximize retrieval coverage.
    Additionally: order subquestions logically if dependencies exist, restrict the number of subquestions to a maximum of 50 and a minimum of 3.
    Give the clear logical and coherent reasoning, in points, leading to the subquestions, before the subquestions block <subquestions></subquestions>.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content":prompt}
            ],
        "temperature":0,
        "top_p":1
        };

    resp = call_with_retry(payload);
    subqs = re.findall(r'<subquestions>(.*?)</subquestions>', resp, re.DOTALL);
    items = []
    for block in subqs:
        lines = block.splitlines()  # split into individual lines
        for line in lines:
            line = line.strip()
            if line:
                items.append(line.lstrip("- ").strip())
    resp = re.findall(r'<reasoning>(.*?)</reasoning>', resp, re.DOTALL);
    return resp, items

def get_embedding(text):
    return model.encode(text).tolist()

def retrieve(query, k=5):
    q_emb = np.array([get_embedding(query)]).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]]

def query_answer_tool(query,subqs,context,model=llmmodel):
    prompt = f"""
    You are given a question, subquestions decomposed from it, and a context.

    Answer ALL PARTS of the question and ALL PARTS of the subquestions COMPLETELY, DIRECTLY and ACCURATELY using the information provided in the CONTEXT.

    RULES:
    - Do NOT use any external knowledge but you are allowed to use your reasoning abilities to combine and synthesize information from the context.
    - You may combine and reason across multiple documents in the context to form the answer.
    - Choose the BEST POSSIBLE ANSWER supported by the context, even if it is not a direct extract from the context.
    - If the question requires comparison or synthesis: 
        + Identify RELEVANT facts from different documents
        + COMBINE them logically
        + DERIVE the final conclusion
    - If the question involves inference, give the facts/points/figures that lead to the inference clearly.
    - If numerical figures are needed, PROVIDE ALL THE NUMBERS from the context and show how you use them to arrive at the answer.
    - Identify KEY QUANTITATIVE FACTS that directly support the answer.
    - The final answer should answer ALL the subquestions and the main question, but should be a logical paragraph of atleast 2 COMPLETE sentences, not a list of sub-answers. 
    - If multiple numbers can answer the question, provide ALL relevant numbers and explain how they contribute to the answer.
    - Missing any compared value makes the answer incomplete.
    - Every claim MUST include a citation in the format (Art. X), right next to the statement made, where X is the document ID.
    - Do NOT write 'doc_id' or any extra text inside citations—only the number.
    - Do NOT provide citations without corresponding statements.
    - If the answer cannot be found in the context, return exactly: [''].

    STRICT REQUIREMENTS (NON-NEGOTIABLE):
    - Your response MUST contain exactly one <answer>...</answer> block.
    - If the <answer> block is missing, your response is INVALID.
    - Do NOT output anything outside the required format.
    - A citation MUST be provided for every claim made just after the claim in the answer
    - ALL subquestions and the main question MUST be answered in a coherent paragraph of atleast 2 sentences, not just listed as separate answers.

    MANDATORY:
    - If the context contains numerical evidence (percentages, counts, statistics), you MUST include them in the answer IF RELEVANT.
    - Especially include key statistics that support the conclusion.    

    FORMAT:
    {{Reasoning...}}

    <answer>final answer with citations</answer>

    If answer not found:
    <answer></answer>

    - The <answer> block must:
      - Contain only the final answer
      - Include all necessary citations
      - Not include reasoning or extra text outside the answer

    QUESTION:
    {query}

    SUBQUESTIONS:
    {subqs}

    CONTEXT:
    {context}
    """
    messages=[
        {"role": "system", "content": "You answer strictly from the provided context."}];
    messages.append({"role": "user", "content": prompt});
    # Add past conversation
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content":prompt}
            ],
        "temperature":0,
        "top_p":1
        };

    resp = call_with_retry(payload);
    if(not resp.strip()):
        prompt = f"""
        Answer the question, to the point, using ONLY the conversation history. Answer in complete sentences enclosed within <answer></answer>. Also, cite the document mentioned in the conversation history. If the answer cannot be found in the conversation history return ONLY a ''."

        Conversation History:
        {conversation_history}

        Question: {query}
        """
        messages=[
            {"role": "system", "content": "You answer strictly from the provided conversational history."}];
        for item in conversation_history:
            for pair in item["subqa"]:
                messages.append({
                    "role": "user",
                    "content": pair["q"]
                })
                messages.append({
                    "role": "assistant",
                    "content": pair["a"]
                })
            messages.append({"role": "user", "content": prompt});
            # Add past conversation
            payload = {
            "model": model,
            "messages": [
                {"role": "user", "content":prompt}
                ],
            "temperature":0,
            "top_p":1
            };

            resp = call_with_retry(payload);
    ans = re.findall(r'<answer>(.*?)</answer>', resp, re.DOTALL);
    if(ans==[]):
        return resp, []
    resp = re.sub(r'<answer>.*?</answer>', '', resp, flags=re.DOTALL)
    return resp, ' '.join(ans)

def main():
# ----------------------------
# Streamlit UI
# ----------------------------
    global API_KEY, model, qa_list, documents, embeddings, conversation_history,llmmodel,index  
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    qa_path = os.path.join(BASE_DIR, "qa_list.json")
    doc_path = os.path.join(BASE_DIR, f"documents_{trans_model}.json")
    
    trans_model = "all_mini_lm"; #bge_large, bge_base, intfloat, all_mini_lm
    model = models["all_mini_lm"]
    API_KEY = st.secrets["API_KEY"];
    model = SentenceTransformer(trans_model);

    with open(qa_path, "r") as f:
        qa_list = json.load(f)
    
    with open(doc_path, "r") as f:
        documents = json.load(f)
    
    embeddings = np.load(f"floats_{trans_model}.npy")
    
    conversation_history = [];
    llmmodel = "meta-llama/llama-3.1-8b-instruct";
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.title("🩺 Healthcare Chat Assistant")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    query = st.text_input("Ask your question:")
    st.markdown("""
    <style>
    /* Remove gap between columns */
    div[data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Remove space between columns container */
    div[data-testid="stHorizontalBlock"] {
        gap: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4,12,26]);
    qnum=0;
    context_str = "";
    with col1:
        if st.button("Ask") and query:
            start_time = time.time()
            query = re.sub(r'\s+', ' ', query).strip()
            conversation_history = st.session_state.history.copy()  # Use session history for the current conversation
            try:
                qnum=st.session_state.history[-1]["qnum"]+1;
            except:
                qnum=0;
            resp, subquestions = query_decomp_tool(query)
            answers = []
            resp2 = [];
            all_queries = [query] + subquestions
    
            contexts = []
            for q in all_queries:
                contexts.append(retrieve(q, k=5))
    
            # flatten AFTER selection
            all_context = [item for sublist in contexts for item in sublist]
    
            # deduplicate
            seen = set()
            filtered_context = []
            for c in all_context:
                key = (c["doc_id"], c["text"])
                if key not in seen:
                    seen.add(key)
                    filtered_context.append(c)
    
            context_str = "\n\n".join([
            f"({c['doc_id']}: {c['title']} {c['text']})\n"
            for c in filtered_context
            ])
            resp2, answers = query_answer_tool(query=query,subqs=subquestions,context=context_str)
    
            if isinstance(resp, list):
                resp = " ".join(resp)
            if isinstance(resp2, list):
                resp2 = ' '.join(resp2);
            steps = resp + "\n" + resp2;
            end_time = time.time()
            # Save conversation
            st.session_state.history.append({
            "time":f"{end_time - start_time:0.2f}",
            "qnum": qnum,
            "question": query,
            "steps": steps,
            "answer":answers
            });
    with col2:
        if(st.button("Clear Conversation")):
            st.session_state.history = [];
    
    grouped = defaultdict(lambda: {"question": "", "steps": "", "answers": []})
    for msg in st.session_state.history:
        qnum = msg["qnum"]
        with st.expander(f"""Reasoning {qnum+1} (Time: {msg['time']}s)""",expanded=False):
            st.write(f"🧠: {msg['steps']}");
        with st.expander(f"""Question {qnum+1}""",expanded=False):
            st.write(f"🧑‍💻: {msg['question']}");
        with st.expander(f"""Answer {qnum+1}""",expanded=False):
            st.write(f"🤖: {msg['answer']}");
            
if __name__ == "__main__":
    main()
