import json
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from together import Together
import fitz  # PyMuPDF
import tarfile
import io
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse

trans_model = "all_mini_lm"

def get_pdf_url(url):
    try:
        # 🔥 Fix abstract URL
        url = url.replace("/abs/", "/")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()

            page = context.new_page()
            page.goto(url, timeout=60000)

            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(3000)

            # 🔥 Capture popup (PDF tab)
            with context.expect_page() as new_page_info:
                page.click("text=View PDF")

            pdf_page = new_page_info.value

            pdf_page.wait_for_load_state()

            pdf_url = pdf_page.url

            browser.close()

            if "pdf.sciencedirectassets.com" in pdf_url:
                print("PDF URL captured ✅")
                return pdf_url
            else:
                print("PDF page opened but URL not correct ❌")
                return None

    except Exception as e:
        print(f"Playwright error: {e}")
        return None


# -----------------------------
# 2. Download PDF
# -----------------------------
def download_pdf(pdf_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.sciencedirect.com/"
        }

        res = requests.get(pdf_url, headers=headers, timeout=20)

        if res.status_code == 200:
            print("PDF downloaded ✅")
            return res.content
        else:
            print("Failed to download PDF ❌")
            return None

    except Exception as e:
        print(f"Download error: {e}")
        return None


# -----------------------------
# 3. Extract text from PDF
# -----------------------------
def extract_text(pdf_bytes):
    try:
        text = ""

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        print("Text extracted ✅")
        return text

    except Exception as e:
        print(f"PDF parse error: {e}")
        return ""


# -----------------------------
# 4. Full pipeline
# -----------------------------
def fetch_sciencedirect_full_text(article_url):
    print(f"\nProcessing: {article_url}")

    pdf_url = get_pdf_url(article_url)

    if not pdf_url:
        return ""

    pdf_bytes = download_pdf(pdf_url)

    if not pdf_bytes:
        return ""

    text = extract_text(pdf_bytes)

    return text


def fetch_with_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, timeout=60000)

            # wait for content to load
            page.wait_for_timeout(5000)

            content = page.content()

            browser.close()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        if len(text) > 500:
            return text

    except Exception as e:
        print(f"Playwright failed: {e}")

    return ""


# -----------------------------
# 1. Check if URL is PMC
# -----------------------------
def is_pmc_url(url):
    return "pmc.ncbi.nlm.nih.gov" in url


# -----------------------------
# 2. Extract PMC ID
# -----------------------------
def extract_pmc_id(url):
    try:
        return url.strip("/").split("/")[-1]
    except:
        return None


# -----------------------------
# 3. Get PMC XML via API
# -----------------------------
def get_pmc_xml(pmc_id):
    try:
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
        res = requests.get(oa_url, timeout=10)

        soup = BeautifulSoup(res.text, "xml")
        link = soup.find("link", {"format": "tgz"})

        if not link:
            return ""

        tgz_url = link["href"]

        if tgz_url.startswith("ftp://"):
            tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov",
                              "https://ftp.ncbi.nlm.nih.gov")

        # Download archive
        res = requests.get(tgz_url, timeout=15)

        # Extract XML
        tar = tarfile.open(fileobj=io.BytesIO(res.content), mode="r:gz")
        for member in tar.getmembers():
            if member.name.endswith(".nxml"):
                f = tar.extractfile(member)
                return f.read().decode("utf-8")

    except Exception as e:
        print(f"PMC fetch failed: {e}")

    return ""


# -----------------------------
# 4. Convert XML to text
# -----------------------------
def parse_pmc_xml(nxml):
    try:
        soup = BeautifulSoup(nxml, "xml")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except:
        return ""


# -----------------------------
# 5. Normal scraping fallback
# -----------------------------
def fetch_html_article(url):
    try:
        headers = {"User-Agent": "Edg/122.0.0.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        return text
    except:
        return ""

def extract_pii(url):
    parts = urlparse(url).path.split('/')
    if 'pii' in parts:
        return parts[parts.index('pii') + 1]
    return None

# -----------------------------
# 6. Unified fetch function
# -----------------------------
def fetch_article(url):
    print(f"\nFetching: {url}")

    # --- PMC route ---
    if is_pmc_url(url):
        pmc_id = extract_pmc_id(url)

        if pmc_id:
            print(f"Detected PMC article: {pmc_id}")

            xml = get_pmc_xml(pmc_id)
            text = parse_pmc_xml(xml)

            if len(text) > 1000:
                print("PMC fetch success ✅")
                return text
            else:
                print("PMC fetch failed or not open access ❌")

    # --- fallback ---
    print("Using HTML fallback...")
    text = fetch_html_article(url)

    # after HTML fallback fails
    if len(text) < 500:
        print("Trying Playwright...")
        text = fetch_with_playwright(url);

        if len(text) > 500:
            print("Playwright success ✅");
            return text

    if(len(text) < 500):
        if "sciencedirect.com" in url or "pubs.acs.org" in url:
            last_part = url.split('/')[-1] + ".pdf";
            with open(last_part, "rb") as f:
                pdf_bytes = f.read()
        return extract_text(pdf_bytes);

    # --- filter bad pages ---
    if "Checking your browser" in text or len(text) < 500:
        print("Blocked or insufficient content ❌")
        return ""

    print("HTML fetch success ✅")
    return text

reader = PdfReader("Together\\healthcare_ai_corpus_v2.pdf")
text = "\n".join([page.extract_text() for page in reader.pages])

pattern = r'(\d{2})\s+[A-Z]+\n(.*?)\n.*?\n(https?://\S+)'

matches = re.findall(pattern, text)

docs_metadata = []
for doc_id, title, url in matches:
    docs_metadata.append({
        "doc_id": int(doc_id),
        "title": title.strip(),
        "url": url.strip()
    })

def chunk_text(text, max_words=50):
    # Step 1: Split into sentences (handles ., ?, !)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_len = len(words)

        # If adding this sentence exceeds limit → start new chunk
        if current_word_count + sentence_len > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_len
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_len

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

model = SentenceTransformer(trans_model)
def get_embedding(text):
    return model.encode(text).tolist()

documents = []
embeddings = []

for doc in docs_metadata:
    article = fetch_article(doc["url"])
    chunks = chunk_text(article,200)
    for chunk in chunks:
        documents.append({
            "text": chunk,
            "doc_id": doc['doc_id'],
            "title": doc["title"],
            "url": doc["url"]
        })
        
        embeddings.append(get_embedding(f"({doc['title']} {chunk})"))

embeddings = np.array(embeddings).astype("float32")

np.save("floats.npy", embeddings)

with open(f"documents_{trans_model}.json", "w") as f:
    json.dump(documents, f)

with open("sentence_transformer_model.txt", "w") as f:
    f.write(trans_model)