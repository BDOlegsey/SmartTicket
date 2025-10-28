# utils/rag.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class TFIDFEmbeddings:
    """Лёгкая замена HuggingFaceEmbeddings на основе TF-IDF."""
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=1024,        # ограничение размерности
            stop_words="english",     # можно заменить на None для русского
            lowercase=True,
            ngram_range=(1, 2)        # unigrams + bigrams
        )
        # Обучаем векторайзер на документах учебника
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def embed_query(self, query: str) -> np.ndarray:
        """Преобразует запрос в вектор."""
        return self.vectorizer.transform([query]).toarray()[0]

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Возвращает k наиболее релевантных фрагментов."""
        query_vec = self.embed_query(query).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]


class RAGPipeline:
    def __init__(self, textbook_text: str):
        # Разбиваем учебник на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(textbook_text)
        self.chunks = chunks
        # Создаём TF-IDF эмбеддинги
        self.embeddings = TFIDFEmbeddings(chunks)

    def retrieve(self, query: str, k: int = 3) -> str:
        results = self.embeddings.similarity_search(query, k=k)
        return "\n\n".join(results)