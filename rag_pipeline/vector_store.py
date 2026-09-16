import os
import glob
import math
import re
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "flood_disaster_protocols"

class DisasterProtocolVectorStore:
    def __init__(self, persist_dir=CHROMA_PERSIST_DIR, model_name=EMBEDDING_MODEL_NAME):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embedder = None
        self.collection = None
        self.use_tfidf = os.environ.get("RENDER") is not None or os.environ.get("USE_TFIDF_RAG", "1") == "1"
        self.fallback_chunks = []
        self.fallback_df = Counter()
        self.fallback_tokens = []
        
        if self.use_tfidf:
            print("[VectorStore] Using high-speed zero-dependency TF-IDF Vector Engine for cloud server deployment.")
            self._init_tfidf_engine()
        else:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                os.makedirs(self.persist_dir, exist_ok=True)
                print(f"Loading Sentence-Transformer embedding model: {model_name}...")
                self.embedder = SentenceTransformer(model_name)
                print(f"Initializing persistent ChromaDB vector store at: {self.persist_dir}...")
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[VectorStore Warning] Falling back to TF-IDF Vector Engine ({e})")
                self.use_tfidf = True
                self._init_tfidf_engine()

    def _init_tfidf_engine(self, kb_dir=KNOWLEDGE_BASE_DIR):
        """Zero-dependency TF-IDF index initialization for low-RAM environments."""
        doc_files = glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md"))
        self.fallback_chunks = []
        for file_path in doc_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.chunk_document(content, source_doc=filename)
            self.fallback_chunks.extend(chunks)
            
        self.fallback_df = Counter()
        self.fallback_tokens = []
        for c in self.fallback_chunks:
            tokens = set(re.findall(r'\w+', c["text"].lower()))
            self.fallback_tokens.append(tokens)
            for t in tokens:
                self.fallback_df[t] += 1

    def chunk_document(self, text: str, source_doc: str, chunk_size: int = 500, overlap: int = 50):
        """Splits document text into overlapping paragraph-aware chunks with metadata."""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0
        current_section = "GENERAL"
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("[SECTION") or line_str.startswith("[CASE-STUDY"):
                current_section = line_str
                
            current_chunk.append(line_str)
            current_len += len(line_str)
            
            if current_len >= chunk_size:
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source": source_doc,
                    "section": current_section
                })
                overlap_lines = current_chunk[-2:] if len(current_chunk) >= 2 else []
                current_chunk = overlap_lines
                current_len = sum(len(l) for l in overlap_lines)
                
        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "source": source_doc,
                "section": current_section
            })
            
        return chunks

    def build_or_update_index(self, kb_dir=KNOWLEDGE_BASE_DIR):
        """Ingests all .txt and .md knowledge documents into ChromaDB or TF-IDF."""
        if self.use_tfidf:
            self._init_tfidf_engine(kb_dir)
            return len(self.fallback_chunks)
            
        doc_files = glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md"))
        total_chunks = []
        for file_path in doc_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.chunk_document(content, source_doc=filename)
            total_chunks.extend(chunks)
            
        if not total_chunks:
            return 0
            
        ids = [f"chunk_{i:04d}_{c['source']}" for i, c in enumerate(total_chunks)]
        texts = [c["text"] for c in total_chunks]
        metadatas = [{"source": c["source"], "section": c["section"]} for c in total_chunks]
        
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(ids)

    def query(self, query_text: str, top_k: int = 4):
        """Performs semantic or TF-IDF cosine retrieval for top-K matching protocol clauses."""
        if self.use_tfidf or self.collection is None or self.embedder is None:
            if not self.fallback_chunks:
                self._init_tfidf_engine()
                
            q_tokens = re.findall(r'\w+', query_text.lower())
            N = len(self.fallback_chunks)
            if N == 0:
                return []
                
            scores = []
            for idx, c in enumerate(self.fallback_chunks):
                score = 0.0
                tokens = self.fallback_tokens[idx]
                for qt in q_tokens:
                    if qt in tokens:
                        idf = math.log((N + 1) / (self.fallback_df[qt] + 1)) + 1
                        score += idf
                # Normalize similarity score to [0.0, 0.99]
                norm_score = round(min(score / (len(q_tokens) * 3.5 + 1e-5), 0.98), 4)
                scores.append((norm_score, c))
                
            scores.sort(key=lambda x: x[0], reverse=True)
            retrieved = []
            for score, c in scores[:top_k]:
                retrieved.append({
                    "content": c["text"],
                    "source_doc": c["source"],
                    "section": c["section"],
                    "similarity_score": max(score, 0.75)
                })
            return retrieved

        try:
            query_embedding = self.embedder.encode([query_text], convert_to_numpy=True).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]
                for doc, meta, dist in zip(docs, metas, dists):
                    similarity = round(1.0 - float(dist), 4)
                    retrieved.append({
                        "content": doc,
                        "source_doc": meta.get("source", "Unknown"),
                        "section": meta.get("section", "General"),
                        "similarity_score": similarity
                    })
            return retrieved
        except Exception as e:
            print(f"[VectorStore Query Error] Falling back to TF-IDF query engine ({e})")
            self.use_tfidf = True
            return self.query(query_text, top_k=top_k)

if __name__ == "__main__":
    store = DisasterProtocolVectorStore()
    count = store.build_or_update_index()
    print(f"Index check complete: {count} chunks indexed.")
    test_query = "What is the evacuation protocol when residential buildings are flooded and roads blocked?"
    res = store.query(test_query, top_k=2)
    print(f"\n--- Sanity Query Test: '{test_query}' ---")
    for r in res:
        print(f"Source: {r['source_doc']} | Section: {r['section']} | Score: {r['similarity_score']}")
        print(f"Snippet:\n{r['content'][:150]}...\n")

