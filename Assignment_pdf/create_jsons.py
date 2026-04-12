import json
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse

pdf_path = "Together\\healthcare_ai_evalset_v2.pdf"
doc = fitz.open(pdf_path)

# Extract full text
text = ""
for page in doc:
    text += page.get_text("text") + "\n"

# Normalize
text = re.sub(r'\r', '', text)

# Split into Q blocks
chunks = re.split(r'(?m)^\s*Q\s*\n?\s*(\d{2})', text)

qa_list = []

# Skip the first element (intro text)
for i in range(1, len(chunks) - 1, 2):

    q_num = chunks[i].strip()
    chunk = chunks[i + 1].strip()

    # ----------------------------
    # 🔥 FILTER OUT FAKE QUESTIONS
    # ----------------------------
    if "Sources:" not in chunk:
        continue

    q_id = "Q" + q_num

    # ----------------------------
    # Remove metadata
    # ----------------------------
    chunk = re.sub(
        r'(EASY|MEDI\s*\n?\s*UM|HAR\s*\n?\s*D|LIVE).*?Sources:.*?\n',
        '',
        chunk,
        flags=re.DOTALL
    ).strip()

    # ----------------------------
    # Extract QUESTION
    # ----------------------------
    if '?' in chunk:
        q_end = chunk.find('?')
        question = chunk[:q_end + 1]
    else:
        lines = chunk.split('\n')

        question_lines = []
        for line in lines:
            line = line.strip()

            if re.match(r'^(\d+%|~?\d+|[A-Z][a-z]+(\s[A-Z][a-z]+)*$)', line):
                break

            question_lines.append(line)

        question = ' '.join(question_lines)

    question = re.sub(r'\s+', ' ', question).strip()

    # ----------------------------
    # Extract ANSWER
    # ----------------------------
    answer = chunk[len(question):].strip()
    answer = re.sub(r'^[\s:–-]+', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    qa_list.append({
        "id": q_id,
        "question": question,
        "answer": answer
    })

with open("qa_list.json", "w") as f:
    json.dump(qa_list, f)