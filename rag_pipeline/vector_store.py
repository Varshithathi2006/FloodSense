"""
FloodSense RAG Vector Store
============================
Manages document chunking, embedding generation using Sentence-Transformers,
and persistent vector indexing with ChromaDB.
"""

import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "flood_disaster_protocols"

class DisasterProtocolVectorStore:
    def __init__(self, persist_dir=CHROMA_PERSIST_DIR, model_name=EMBEDDING_MODEL_NAME):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        print(f"Loading Sentence-Transformer embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        print(f"Initializing persistent ChromaDB vector store at: {self.persist_dir}...")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

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
                # Keep last overlap lines
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
        """Ingests all .txt and .md knowledge documents into ChromaDB."""
        doc_files = glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md"))
        print(f"Found {len(doc_files)} knowledge base documents in {kb_dir}...")
        
        total_chunks = []
        for file_path in doc_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.chunk_document(content, source_doc=filename)
            total_chunks.extend(chunks)
            print(f"  • {filename} -> {len(chunks)} chunks")
            
        if not total_chunks:
            print("No chunks generated. Check knowledge base directory.")
            return 0
            
        # Extract texts, metadata and IDs
        ids = [f"chunk_{i:04d}_{c['source']}" for i, c in enumerate(total_chunks)]
        texts = [c["text"] for c in total_chunks]
        metadatas = [{"source": c["source"], "section": c["section"]} for c in total_chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
        
        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(ids)} document chunks into ChromaDB collection '{COLLECTION_NAME}'!\n")
        return len(ids)

    def query(self, query_text: str, top_k: int = 4):
        """Performs cosine semantic retrieval for top-K matching protocol clauses."""
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
                # Cosine similarity = 1.0 - distance
                similarity = round(1.0 - float(dist), 4)
                retrieved.append({
                    "content": doc,
                    "source_doc": meta.get("source", "Unknown"),
                    "section": meta.get("section", "General"),
                    "similarity_score": similarity
                })
        return retrieved

if __name__ == "__main__":
    store = DisasterProtocolVectorStore()
    count = store.build_or_update_index()
    print(f"Index check complete: {count} chunks indexed.")
    
    # Quick sanity query
    test_query = "What is the evacuation protocol when residential buildings are flooded and roads blocked?"
    res = store.query(test_query, top_k=2)
    print(f"\n--- Sanity Query Test: '{test_query}' ---")
    for r in res:
        print(f"Source: {r['source_doc']} | Section: {r['section']} | Score: {r['similarity_score']}")
        print(f"Snippet:\n{r['content'][:150]}...\n")
