# analyzer_agent_improved.py
import logging
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AnalyzerInput(BaseModel):
    """Входные данные для AnalyzerAgent"""
    text: str
    chunk_size: int = 1000
    overlap: int = 100


class AnalyzerMetrics(BaseModel):
    """Метрики AnalyzerAgent"""
    success_rate: float
    embedding_quality: float
    text_completeness: float
    chunks_count: int
    avg_chunk_size: float


class AnalyzerAgent:
    """Анализирует текст и создает embeddings"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=768, ngram_range=(1, 2))
        self.embeddings = None
        self.chunks = []
        logger.info("✓ AnalyzerAgent инициализирован")

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Разбивает текст на chunks с перекрытием"""
        chunks = []

        # Разбить на предложения для лучшей семантики
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]

        current_chunk = ""
        for sentence in sentences:
            # Если добавление предложения превышает размер, сохраняем chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Перекрытие - начнем следующий chunk с последних 100 символов
                current_chunk = current_chunk[-overlap:] + " " + sentence + "."
            else:
                current_chunk += " " + sentence + "."

        # Добавить последний chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Фильтр: исключить очень маленькие chunks
        self.chunks = [c for c in chunks if len(c) >= 200]

        logger.info(f"✓ Текст разбит на {len(self.chunks)} chunks")
        logger.info(f"  Средний размер: {np.mean([len(c) for c in self.chunks]):.0f} символов")

        return self.chunks

    def create_embeddings(self, chunks: List[str] = None) -> np.ndarray:
        """Создает embeddings для chunks используя TF-IDF"""
        if chunks is None:
            chunks = self.chunks

        if not chunks:
            logger.warning("⚠ Нет chunks для embedding")
            return np.array([])

        try:
            # Создать TF-IDF embeddings (768-dimensional vectors)
            self.embeddings = self.vectorizer.fit_transform(chunks).toarray()

            logger.info(f"✓ Создано {len(self.embeddings)} embeddings (размер: {self.embeddings.shape})")

            return self.embeddings

        except Exception as e:
            logger.error(f"❌ Ошибка при создании embeddings: {e}")
            self.embeddings = np.array([])
            return self.embeddings

    def analyze_text(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Полный анализ текста"""
        chunks = self.split_text(text)
        embeddings = self.create_embeddings(chunks)
        return chunks, embeddings

    def compute_metrics(self) -> AnalyzerMetrics:
        """Вычисляет метрики качества"""

        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("⚠ Нет данных для вычисления метрик")
            return AnalyzerMetrics(
                success_rate=0.0,
                embedding_quality=0.0,
                text_completeness=0.0,
                chunks_count=len(self.chunks),
                avg_chunk_size=0.0
            )

        # Success Rate: процент chunks размером > 200 символов
        large_chunks = sum(1 for c in self.chunks if len(c) > 200)
        success_rate = (large_chunks / len(self.chunks) * 100) if self.chunks else 0.0

        # Embedding Quality: среднее косинусное сходство между соседними chunks
        embedding_quality = self._compute_embedding_quality()

        # Text Completeness: процент сохраненного текста от исходного
        total_original = sum(len(c) for c in self.chunks) + 200  # +200 на разделители
        text_completeness = (sum(len(c) for c in self.chunks) / total_original * 100) if total_original > 0 else 0.0

        # Chunk Statistics
        avg_chunk_size = np.mean([len(c) for c in self.chunks]) if self.chunks else 0.0

        metrics = AnalyzerMetrics(
            success_rate=min(100.0, success_rate),
            embedding_quality=min(1.0, embedding_quality),  # Normalize to 0-1
            text_completeness=min(100.0, text_completeness),
            chunks_count=len(self.chunks),
            avg_chunk_size=float(avg_chunk_size)
        )

        logger.info(f"✓ Метрики AnalyzerAgent вычислены:")
        logger.info(f"  Success Rate: {metrics.success_rate:.1f}%")
        logger.info(f"  Embedding Quality: {metrics.embedding_quality:.3f}")
        logger.info(f"  Text Completeness: {metrics.text_completeness:.1f}%")

        return metrics

    def _compute_embedding_quality(self) -> float:
        """Вычисляет качество embeddings на основе косинусного сходства"""
        if self.embeddings is None or len(self.embeddings) < 2:
            return 0.0

        try:
            # Вычислить матрицу сходства
            similarity_matrix = cosine_similarity(self.embeddings)

            # Взять среднее сходство между соседними chunks
            diagonal_1 = np.diag(similarity_matrix, k=1)

            if len(diagonal_1) == 0:
                return 0.5

            # Фильтр: убрать нулевые значения
            valid_similarities = diagonal_1[diagonal_1 > 0]

            if len(valid_similarities) == 0:
                return 0.3  # Fallback

            quality = np.mean(valid_similarities)

            return float(max(0.0, min(1.0, quality)))

        except Exception as e:
            logger.warning(f"⚠ Ошибка при вычислении качества embeddings: {e}")
            return 0.3


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Пример текста
    text = """
    Искусственный интеллект это область информатики, которая занимается созданием 
    интеллектуальных систем. Машинное обучение является подразделом искусственного интеллекта.
    Нейронные сети моделируют работу мозга человека. Глубокое обучение использует многослойные 
    сети для решения сложных задач. Естественная обработка языка позволяет компьютерам понимать 
    текст. Компьютерное зрение позволяет системам видеть и анализировать изображения.
    """

    analyzer = AnalyzerAgent()
    chunks, embeddings = analyzer.analyze_text(text)
    metrics = analyzer.compute_metrics()

    print(f"\nЧунков: {metrics.chunks_count}")
    print(f"Success Rate: {metrics.success_rate:.1f}%")
    print(f"Embedding Quality: {metrics.embedding_quality:.3f}")
    print(f"Text Completeness: {metrics.text_completeness:.1f}%")
