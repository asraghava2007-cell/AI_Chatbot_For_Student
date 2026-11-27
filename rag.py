import os
import re
from typing import List, Dict, Any, Optional
import numpy as np

# PyPDF for reading PDF files
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# TF-IDF Vector Space Model (Industry-standard Information Retrieval)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine_similarity
except Exception:
    TfidfVectorizer = None
    tfidf_cosine_similarity = None


def extract_text_from_file(file_path_or_bytes, filename: str) -> str:
    """Extracts raw text from .txt, .md, or .pdf files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".txt", ".md"]:
        if isinstance(file_path_or_bytes, (str, os.PathLike)):
            with open(file_path_or_bytes, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        else:
            return file_path_or_bytes.read().decode("utf-8", errors="replace")

    elif ext == ".pdf":
        if PdfReader is None:
            return "Error: pypdf is not installed. Please install pypdf to read PDF files."
        try:
            reader = PdfReader(file_path_or_bytes)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"--- [Page {i + 1}] ---\n{text}")
            return "\n\n".join(pages_text)
        except Exception as e:
            return f"Error reading PDF {filename}: {e}"

    return ""


def chunk_text(text: str, source_name: str, chunk_size: int = 1400, chunk_overlap: int = 250) -> List[Dict[str, Any]]:
    """
    Intelligently splits text into semantic chunks for study materials and question banks.
    Preserves question-answer blocks, sections, days of the week, and paragraph boundaries.
    """
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\r\n", "\n", text).strip()

    # Split on double newlines or question bank/section boundaries (e.g., Q1, Question, Module, Part A, Days)
    sections = re.split(
        r"\n{2,}|(?=\b(?:QUESTION\s+\d+|Q\d+[\.:\)]|MODULE\s+\d+|PART\s+[A-Z]|UNIT\s+\d+|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|PERIOD TIMINGS|COURSES AND FACULTY|ATTENDANCE|ID CARD|FEES|INTERNAL ASSESSMENT)\b)",
        cleaned,
        flags=re.IGNORECASE
    )
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    chunk_index = 1
    current_chunk = ""

    for sec in sections:
        if len(current_chunk) + len(sec) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{sec}".strip()
        else:
            if current_chunk:
                chunks.append({
                    "source": source_name,
                    "chunk_id": f"{source_name}_chunk_{chunk_index}",
                    "chunk_num": chunk_index,
                    "content": current_chunk
                })
                chunk_index += 1
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = f"{overlap_text}\n\n{sec}".strip()
            else:
                # Long section: split by lines/paragraphs
                lines = sec.split("\n")
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) + 1 <= chunk_size:
                        sub_chunk = f"{sub_chunk}\n{line}".strip()
                    else:
                        if sub_chunk:
                            chunks.append({
                                "source": source_name,
                                "chunk_id": f"{source_name}_chunk_{chunk_index}",
                                "chunk_num": chunk_index,
                                "content": sub_chunk
                            })
                            chunk_index += 1
                        sub_chunk = line
                if sub_chunk:
                    chunks.append({
                        "source": source_name,
                        "chunk_id": f"{source_name}_chunk_{chunk_index}",
                        "chunk_num": chunk_index,
                        "content": sub_chunk
                    })
                    chunk_index += 1
                current_chunk = ""

    if current_chunk:
        chunks.append({
            "source": source_name,
            "chunk_id": f"{source_name}_chunk_{chunk_index}",
            "chunk_num": chunk_index,
            "content": current_chunk
        })

    return chunks


class StudentRAG:
    """
    Retrieval-Augmented Generation (RAG) Engine for Student Query Assistant:
    - High-precision TF-IDF Vector Space Model (Unigrams + Bigrams, Sublinear Term Frequency)
    - Cosine Similarity Matching over Document Vectors
    - Dynamic Multi-Format Document Ingestion (PDF, TXT, MD)
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def add_document(self, content_or_stream, filename: str) -> int:
        """Extracts text, segments into chunks, and updates the vector index."""
        text = extract_text_from_file(content_or_stream, filename)
        if not text:
            return 0
        new_chunks = chunk_text(text, source_name=filename)
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        self._reindex()
        return len(new_chunks)

    def add_raw_text(self, text: str, source_name: str) -> int:
        """Indexes preloaded raw text documents (timetable, college info)."""
        if not text or not text.strip():
            return 0
        new_chunks = chunk_text(text, source_name=source_name)
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        self._reindex()
        return len(new_chunks)

    def _reindex(self):
        """Builds TF-IDF vector matrix across all document chunks."""
        if not self.chunks:
            self.tfidf_matrix = None
            return

        corpus = [c["content"] for c in self.chunks]

        if TfidfVectorizer is not None:
            try:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english"
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            except Exception as e:
                print(f"[RAG] TF-IDF indexing error: {e}")
                self.tfidf_matrix = None

    def _expand_query(self, query: str) -> str:
        """Lightweight semantic query expansion for college domain terminology."""
        synonym_map = {
            "teacher": "faculty professor instructor",
            "teachers": "faculty professors instructors",
            "teach": "faculty",
            "teaches": "faculty instructor",
            "prof": "faculty professor",
            "sub": "course subject",
            "subs": "courses subjects",
            "subject": "course subject",
            "subjects": "courses subjects",
            "test": "internal assessment IA exam",
            "tests": "internal assessment IA exam",
            "exam": "internal assessment IA exam",
            "exams": "internal assessment IA exam",
            "time": "period timings schedule",
            "timings": "period timings schedule",
            "timing": "period timings schedule",
            "leave": "medical leave attendance",
            "id": "ID card library",
        }
        words = query.lower().split()
        expanded_terms = []
        for w in words:
            clean_w = re.sub(r"[^\w]", "", w)
            if clean_w in synonym_map:
                expanded_terms.append(synonym_map[clean_w])
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using Cosine Similarity.
        Includes document-order reassembly and context window expansion for full answers.
        """
        if not self.chunks or not query or not query.strip():
            return []

        expanded_query = self._expand_query(query)
        results = []

        # Vector Space Retrieval (TF-IDF Cosine Similarity)
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([expanded_query])
                scores = tfidf_cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                
                # Filter out chunks with 0 score, sort descending
                non_zero_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]
                selected_indices = set(non_zero_indices[:top_k])

                # Context Stitching: For top matching chunks, also pull subsequent adjacent chunk
                # so multi-part questions (Part a, b, c) are not truncated mid-sentence
                for idx in list(selected_indices)[:3]:
                    current_chunk = self.chunks[idx]
                    if "chunk_num" in current_chunk:
                        next_num = current_chunk["chunk_num"] + 1
                        # Find if adjacent chunk exists in the same source
                        for other_idx, other_chunk in enumerate(self.chunks):
                            if (other_chunk.get("source") == current_chunk.get("source") and
                                other_chunk.get("chunk_num") == next_num):
                                selected_indices.add(other_idx)
                                break

                for idx in selected_indices:
                    chunk = self.chunks[idx].copy()
                    chunk["score"] = float(scores[idx])
                    results.append(chunk)

                # Order chunks by source and chunk_num so reading flow is natural
                results.sort(key=lambda x: (x.get("source", ""), x.get("chunk_num", 0)))
                return results
            except Exception as e:
                print(f"[RAG] TF-IDF search error: {e}")

        # Fallback: Keyword Overlap
        keywords = set(re.findall(r"\w+", expanded_query.lower()))
        scored = []
        for c in self.chunks:
            c_words = set(re.findall(r"\w+", c["content"].lower()))
            overlap = len(keywords & c_words)
            chunk = c.copy()
            chunk["score"] = float(overlap)
            scored.append(chunk)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
