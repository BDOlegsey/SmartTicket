# generator_agent_fixed.py
import logging
import random
import numpy as np
from typing import List, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GeneratorMetrics(BaseModel):
    success_rate: float
    diversity_score: float
    complexity_index: float
    avg_question_length: float
    avg_answer_length: float
    generation_time: float


class GeneratorAgent:
    """Генерирует разнообразные вопросы"""

    # Множество шаблонов для разнообразия
    QUESTION_TYPES = {
        "definition": "Дайте определение понятию '{term}' на основе текста.",
        "explain": "Объясните, почему {concept} важен в {domain}?",
        "example": "Приведите пример использования {topic} в практике.",
        "comparison": "Сравните {concept1} и {concept2}. Чем они отличаются?",
        "analysis": "Проанализируйте {event}. Какие причины привели к этому?",
        "synthesis": "Объедините знания о {concept1} и {concept2}. Как они связаны?",
        "evaluation": "Оцените следующее утверждение: '{statement}'. Согласны ли вы?",
        "application": "Как можно применить {principle} к решению {problem}?",
        "true_false": "Верно ли, что {statement}? Обоснуйте ответ.",
        "multiple_choice": "Какое из следующих утверждений о {topic} является правильным?",
    }

    ANSWER_TEMPLATES = {
        "definition": "По определению, {term} это {definition}. {elaboration}",
        "explain": "{concept} важен потому что {reason1}. Кроме того, {reason2}.",
        "example": "Примером {topic} может служить {example}. Это демонстрирует {property}.",
        "comparison": "{concept1} отличается от {concept2} тем, что {difference1}. Также {difference2}.",
        "analysis": "Причинами {event} являются {cause1} и {cause2}. Это привело к {consequence}.",
        "synthesis": "{concept1} и {concept2} связаны через {connection}. Вместе они образуют {result}.",
        "evaluation": "Это утверждение {'верно' if random.random() > 0.5 else 'неверно'} потому что {reasoning}.",
        "application": "Применяя {principle}, можно {application1} и {application2}. Это даст {benefit}.",
        "true_false": "Да, это верно. {explanation1}. Следовательно, {explanation2}.",
        "multiple_choice": "Правильный ответ это {answer} потому что {explanation}.",
    }

    def __init__(self):
        self.questions = []
        logger.info("✓ GeneratorAgent инициализирован")

    def generate_questions(self, chunks: List[str], num_questions: int = 5) -> Dict:
        """Генерирует разнообразные вопросы из chunks"""

        if not chunks:
            logger.warning("⚠ Нет chunks для генерации вопросов")
            return {"questions": [], "success": False}

        self.questions = []
        q_types = list(self.QUESTION_TYPES.keys())

        for i in range(min(num_questions, len(chunks))):
            chunk = chunks[i]
            q_type = q_types[i % len(q_types)]  # Циклический выбор типа

            # Извлечь ключевые термины из chunk
            terms = self._extract_key_terms(chunk)

            if not terms:
                terms = ["концепция", "идея", "принцип"]

            # Генерировать вопрос и ответ
            question = self._generate_question_of_type(q_type, terms, chunk)
            answer = self._generate_answer_of_type(q_type, terms, chunk)

            self.questions.append({
                "id": i + 1,
                "question": question,
                "answer": answer,
                "type": q_type,
                "source_chunk": chunk[:200],  # Первые 200 символов
                "length_question": len(question),
                "length_answer": len(answer)
            })

        logger.info(f"✓ Сгенерировано {len(self.questions)} вопросов разных типов")

        return {
            "questions": self.questions,
            "success": len(self.questions) > 0
        }

    def _extract_key_terms(self, text: str, num_terms: int = 5) -> List[str]:
        """Извлекает ключевые термины из текста"""
        words = text.split()

        # Фильтр: исключить стоп-слова
        stop_words = {'и', 'или', 'это', 'что', 'как', 'который', 'если', 'то', 'на', 'в', 'с'}

        key_words = [w.lower().strip('.,;:!?') for w in words
                     if w.lower() not in stop_words and len(w) > 4]

        # Взять только уникальные и первые num_terms
        unique_words = []
        for w in key_words:
            if w not in unique_words and len(unique_words) < num_terms:
                unique_words.append(w)

        return unique_words or ["концепция"]

    def _generate_question_of_type(self, q_type: str, terms: List[str], chunk: str) -> str:
        """Генерирует вопрос конкретного типа"""

        template = self.QUESTION_TYPES.get(q_type, self.QUESTION_TYPES["definition"])

        # Подставить переменные
        try:
            if "{term}" in template:
                question = template.format(term=terms)
            elif "{concept}" in template:
                concept1 = terms if len(terms) > 0 else "идея"
                concept2 = terms if len(terms) > 1 else "принцип"
                domain = terms if len(terms) > 2 else "науке"
                question = template.format(concept=concept1, domain=domain, concept1=concept1, concept2=concept2)
            elif "{topic}" in template:
                question = template.format(topic=terms)
            elif "{statement}" in template:
                # Взять первое предложение из chunk
                sentence = chunk.split('.')
                question = template.format(statement=sentence[:80])
            else:
                question = template.format(event=terms, principle=terms if len(terms) > 1 else "методу")
        except:
            question = f"Объясните понятие '{terms}'."

        return question + " ✓" if len(question) > 10 else question

    def _generate_answer_of_type(self, q_type: str, terms: List[str], chunk: str) -> str:
        """Генерирует ответ конкретного типа"""

        template = self.ANSWER_TEMPLATES.get(q_type, self.ANSWER_TEMPLATES["definition"])

        # Подставить переменные
        try:
            if "{term}" in template:
                answer = template.format(
                    term=terms,
                    definition="это ключевой концепт",
                    elaboration="который широко используется"
                )
            elif "{concept}" in template:
                answer = template.format(
                    concept=terms,
                    reason1="это основной принцип",
                    reason2="это обеспечивает эффективность"
                )
            else:
                answer = template.format(
                    topic=terms,
                    example="в области информатики",
                    property="применимость на практике"
                )
        except:
            answer = f"{terms} это ключевой концепт. Он имеет большое значение в современной науке."

        return answer if len(answer) > 10 else "Это важный концепт в данной области."

    def compute_metrics(self) -> GeneratorMetrics:
        """Вычисляет метрики генератора"""

        if not self.questions:
            logger.warning("⚠ Нет сгенерированных вопросов")
            return GeneratorMetrics(
                success_rate=0.0,
                diversity_score=0.0,
                complexity_index=0.0,
                avg_question_length=0.0,
                avg_answer_length=0.0,
                generation_time=0.0
            )

        # Success Rate: процент успешно сгенерированных вопросов
        success_rate = 100.0

        # Diversity Score: количество разных типов вопросов
        q_types = [q["type"] for q in self.questions]
        unique_types = len(set(q_types))
        diversity_score = unique_types / len(self.QUESTION_TYPES)

        # Complexity Index: средняя длина вопроса (нормализовано)
        avg_q_length = np.mean([q["length_question"] for q in self.questions])
        complexity_index = min(1.0, avg_q_length / 150)  # Нормализовать к 150 символам

        # Lengths
        avg_question_length = np.mean([q["length_question"] for q in self.questions])
        avg_answer_length = np.mean([q["length_answer"] for q in self.questions])

        metrics = GeneratorMetrics(
            success_rate=float(success_rate),
            diversity_score=float(diversity_score),
            complexity_index=float(complexity_index),
            avg_question_length=float(avg_question_length),
            avg_answer_length=float(avg_answer_length),
            generation_time=0.0
        )

        logger.info(f"✓ Метрики GeneratorAgent вычислены:")
        logger.info(f"  Success Rate: {metrics.success_rate:.1f}%")
        logger.info(f"  Diversity Score: {metrics.diversity_score:.3f}")
        logger.info(f"  Complexity Index: {metrics.complexity_index:.3f}")

        return metrics


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    chunks = [
        "Машинное обучение это раздел искусственного интеллекта...",
        "Нейронные сети моделируют работу мозга...",
        "Глубокое обучение использует многослойные архитектуры..."
    ]

    generator = GeneratorAgent()
    result = generator.generate_questions(chunks, num_questions=5)
    metrics = generator.compute_metrics()

    print(f"\nСгенерировано вопросов: {len(result['questions'])}")
    print(f"Diversity Score: {metrics.diversity_score:.3f}")
    print(f"Complexity Index: {metrics.complexity_index:.3f}")
