import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import requests

trans_model = "all_mini_lm";#bge_large, bge_small, intfloat, all_mini_lm
model = SentenceTransformer(trans_model);
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

with open("qa_list.json", "r") as f:
    qa_list = json.load(f)

with open(f"documents_{trans_model}.json", "r") as f:
    documents = json.load(f)

embeddings = np.load("floats.npy")

conversation_history = [];
llmmodel = "meta-llama/llama-3.1-8b-instruct";
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

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

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
    """
    prompt = f
    Answer the question {query}, completely and to the point, using only the context:
    {context}.
    Exhaustively search the context for relevant information to give the most suitable answer.
    Answer providing all the details and enclose it within <answer>ans</answer> ONLY where 'ans' is the answer to the query. 
    The <answer>ans</answer> block should also include the citation of the document(s) from which the answer has been extracted. 
    The citation MUST be given for every claim made by giving the ID of the document from which the claim was taken, 
    specified in the format (Art. doc_id). DON'T print 'doc_id'. doc_id should be just a number without any other text.
    DON'T give just the citation.
    You MUST provide a clear, logical, and coherent reasoning, before the <answer></answer> block, aligned with the answer.
    If the answer cannot be found from the context, just return an empty string [''].
    The final response should include ONLY the answer within the block <answer></answer> preceded by the reasoning. There should be NO other text.
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

score = [""]*11;
idx = 0;
for objs in qa_list[7:]:
    query = objs["question"];
    exp_ans = objs["answer"];
    resp, subquestions = query_decomp_tool(query)
    answers = []
    resp2 = [];
    all_queries = [query] + subquestions

    contexts = []
    for q in all_queries:
        dense = retrieve(q, k=15)
        #sparse = keyword_match(q, documents)
        contexts.append(dense)

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
    """
    def rerank(query, docs):
        scored = []
        for d in docs:
            score = max([
                np.dot(get_embedding(q), get_embedding(d["text"]))
                for q in all_queries
            ])
            scored.append((score, d))
        scored.sort(reverse=True)
        return [d for _, d in scored[:20]]
    filtered_context = rerank(query, filtered_context)

    filtered_context = filtered_context[:15]
    """
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
    prompt = f"""
    You are an objective evaluator for a question-answering system. Return JUST the total score and no other text, an integer between 0 and 10, 
    in the end of the response, enclosed within the tags <total_score>{{s}}</total_score> where 's' is the score.

    Your task is to score the model-generated answer against the expected (ground truth) answer.

    Inputs:
    - Question = {query}
    - Model Answer = {answers}
    - Expected Answer = {exp_ans}

    Evaluate the answer using the following criteria:

    1. Factual Accuracy (0–3)
        - 3: All key facts match the expected answer.
        - 2: Mostly correct with minor inaccuracies.
        - 1: Significant errors or missing key facts.
        - 0: Incorrect or misleading.

        Note: Do NOT penalize for differences in wording if the meaning is equivalent.

    2. Citation Quality (0–3)
        - 3: Citations are present and support the claims.
        - 2: Citations are present but partially relevant or slightly incorrect.
        - 1: Citations are weak, incomplete, or poorly formatted.
        - 0: No citations or completely irrelevant citations.

    3. Reasoning Trace (0–2) given by {steps}
        - 2: Clear, logical, and coherent reasoning aligned with the answer.
        - 1: Partial or somewhat unclear reasoning.
        - 0: No reasoning or incoherent reasoning.

    4. Completeness (0–2)
        - 2: Covers all major aspects of the expected answer.
        - 1: Covers some aspects but misses important points.
        - 0: Very incomplete.

    Instructions:
    - Be strict but fair.
    - Base your evaluation only on the provided inputs.
    - Do not hallucinate missing information.
    - If reasoning steps are not provided, assign 0 for Reasoning Trace.

    """
    temp="""
    Output format (strictly follow this):
    {{
        "factual_accuracy": <0-3>,
        "citation_quality": <0-3>,
        "reasoning_trace": <0-2>,
        "completeness": <0-2>,
        "total_score": <0-10>,
        "justification": "<brief explanation for the scores>"
    }}
    """
    resp = call_with_retry(
    payload={
        "model": llmmodel,
        "messages": [
            {"role": "user", "content":prompt}
            ],
        "temperature":0,
        "top_p":1
        });
    """
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
        }, json=payload)
        """
    score[idx] = re.findall(r'<total_score>(.*?)</total_score>', resp, re.DOTALL)
    print(answers);
    print(exp_ans);
    print(score[idx]);
    print("\n");
    if(score[idx]==[]):
        score[idx] = "0";
    idx+=1;