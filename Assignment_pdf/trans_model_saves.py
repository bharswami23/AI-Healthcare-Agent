from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("all_mini_lm")

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model.save("bge_base")

model = SentenceTransformer("intfloat/e5-base-v2")
model.save("intfloat")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
model.save("bge_large")