from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = None

class SentencePair(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/similarity")
def get_similarity(pair: SentencePair):
    global model
    if model is None:
        print("🔄 Cargando modelo MiniLM por primera vez...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Modelo cargado")
    emb1 = model.encode(pair.sentence1, convert_to_tensor=True)
    emb2 = model.encode(pair.sentence2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return {"similarity_score": round(similarity, 4)}

@app.get("/")
def root():
    return {"message": "API de similitud semántica operativa"}
