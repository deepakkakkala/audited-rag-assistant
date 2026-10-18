import os
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
import google.generativeai as genai

load_dotenv()

# Configure API Key 
# use os to get the api key in the .env file
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "audited_knowledge_base"
COSINE_DISTANCE_THRESHOLD = 0.99  # Cosine distance: 0 is exact match, 1 is orthogonal
# using 0.55 as a threshold for the similarity
class Citation(BaseModel):
    source_file: str = Field(description="Name of the source file")
    quoted_snippet: str = Field(description="Direct phrase or excerpt supporting the statement")

class AuditedResponse(BaseModel):
    answer: str = Field(description="Direct, precise answer synthesized only from provided context")
    citations: List[Citation] = Field(description="List of exact citations and references used")
    confidence_rationale: str = Field(description="Brief explanation of why this context fulfills the query")

class RAGEngine:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn
        )
        self.model = genai.GenerativeModel("gemini-3.8-flash")

    def query(self, user_query: str) -> dict:
        # Retrieve top 3 chunks with distances
        results = self.collection.query(
            query_texts=[user_query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return {
                "status": "refused",
                "answer": "No relevant documents were found in the knowledge base.",
                "citations": [],
                "best_distance": 1.0
            }

        best_distance = distances[0]

        # Guardrail Check: Block retrieval if context distance is too high
        if best_distance > COSINE_DISTANCE_THRESHOLD:
            return {
                "status": "refused",
                "answer": f"Context relevance guardrail triggered (Distance: {best_distance:.3f} > Threshold: {COSINE_DISTANCE_THRESHOLD}). The query is out-of-scope or not sufficiently supported by the documentation.",
                "citations": [],
                "best_distance": best_distance
            }

        # Build Context Block
        context_blocks = []
        for doc, meta in zip(docs, metadatas):
            context_blocks.append(f"--- SOURCE: {meta['source']} ---\n{doc}")
        formatted_context = "\n\n".join(context_blocks)

        prompt = f"""
You are an Audited AI Verification Assistant. Answer the question using ONLY the context provided below.
Do not assume, extrapolate, or use external knowledge. 
If the context does not explicitly verify the answer, state that you cannot answer.

Context:
{formatted_context}

Question: {user_query}

You must return a valid JSON object matching this schema:
{{
  "answer": "string",
  "citations": [
    {{
      "source_file": "string",
      "quoted_snippet": "string"
    }}
  ],
  "confidence_rationale": "string"
}}
Return ONLY JSON. Do not wrap in markdown fences or write introductory text.
"""

        response = self.model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Strip potential markdown formatting if returned
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        try:
            parsed_data = json.loads(raw_text)
            validated = AuditedResponse(**parsed_data)
            return {
                "status": "verified",
                "answer": validated.answer,
                "citations": [c.model_dump() for c in validated.citations],
                "confidence_rationale": validated.confidence_rationale,
                "best_distance": best_distance,
                "retrieved_chunks": docs
            }
        except Exception as e:
            return {
                "status": "format_error",
                "answer": raw_text,
                "citations": [],
                "best_distance": best_distance,
                "error": str(e)
            }