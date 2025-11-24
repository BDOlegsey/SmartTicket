import json
import re
from typing import Literal
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# 1. PYDANTIC МОДЕЛИ (структуры данных)
# ============================================================================

class VerificationIssue(BaseModel):
    """Проблема, найденная при верификации"""
    type: Literal[
        "factual_error",
        "incomplete_answer",
        "irrelevant_context",
        "too_complex",
        "duplicated_question"
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    suggested_fix: str


class Question(BaseModel):
    """Входной вопрос от Генератора"""
    id: int
    question_text: str
    answer: str


class VerifiedQuestion(BaseModel):
    """Проверенный вопрос с результатами верификации"""
    id: int
    question_text: str
    answer: str
    verification_status: Literal["approved", "needs_revision", "rejected"]
    issues: list[VerificationIssue] = Field(default_factory=list)
    source_chunks_ids: list[int] = Field(default_factory=list)
    feedback: str = ""


class VerifierOutput(BaseModel):
    """Выходные данные верификации"""
    status: Literal["success", "error"]
    verified_questions: list[VerifiedQuestion] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    error_message: str | None = None


# ============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def calculate_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Вычисляет косинусное подобие между двумя эмбедингами"""
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)


def find_relevant_chunks(
    question_embedding: list[float],
    chunk_embeddings: list[list[float]],
    threshold: float = 0.6,
    top_k: int = 5
) -> tuple[list[int], float]:
    """
    Находит релевантные чанки по эмбедингам
    
    Returns:
        Кортеж (список ID релевантных чанков, среднее подобие)
    """
    if not chunk_embeddings:
        return [], 0.0
    
    question_emb = np.array(question_embedding).reshape(1, -1)
    chunk_embs = np.array(chunk_embeddings)
    similarities = cosine_similarity(question_emb, chunk_embs)[0]
    
    # Фильтруем по threshold
    relevant_indices = np.where(similarities >= threshold)[0]
    
    if len(relevant_indices) == 0:
        return [], float(np.max(similarities) if len(similarities) > 0 else 0)
    
    # Берём top-k наиболее подобные
    top_indices = np.argsort(similarities[relevant_indices])[-top_k:][::-1]
    chunk_ids = [int(idx) for idx in relevant_indices[top_indices]]
    avg_similarity = float(np.mean(similarities[relevant_indices]))
    
    return chunk_ids, avg_similarity


def check_duplication(
    current_embedding: list[float],
    previous_embeddings: dict[int, list[float]],
    duplication_threshold: float = 0.9
) -> list[int]:
    """Проверяет дублирование вопроса с предыдущими"""
    duplicates = []
    
    for q_id, prev_emb in previous_embeddings.items():
        similarity = calculate_similarity(current_embedding, prev_emb)
        if similarity > duplication_threshold:
            duplicates.append(q_id)
    
    return duplicates


def get_question_embedding(
    question_text: str,
    embeddings_model
) -> list[float]:
    """Получает эмбединг вопроса"""
    return embeddings_model.embed_query(question_text)


# ============================================================================
# 3. ОСНОВНЫЕ ФУНКЦИИ ВЕРИФИКАЦИИ
# ============================================================================

def check_relevance(
    question: Question,
    text_chunks: list[str],
    chunk_embeddings: list[list[float]],
    embeddings_model,
    threshold_similarity: float = 0.6
) -> tuple[list[int], bool, list[float]]:
    """
    ПРОВЕРКА 1: Релевантность вопроса источнику
    
    Returns:
        Кортеж (ID релевантных чанков, релевантен ли вопрос, embedding вопроса)
    """
    question_embedding = get_question_embedding(question.question_text, embeddings_model)
    relevant_chunks, avg_sim = find_relevant_chunks(
        question_embedding,
        chunk_embeddings,
        threshold=threshold_similarity,
        top_k=5
    )
    
    is_relevant = len(relevant_chunks) > 0
    return relevant_chunks, is_relevant, question_embedding


def check_answer_correctness(
    question: Question,
    text_chunks: list[str],
    relevant_chunk_ids: list[int],
    llm
) -> list[VerificationIssue]:
    """
    ПРОВЕРКА 2: Корректность ответа через LLM
    
    Использует LLM для анализа ответа против контекста учебника
    """
    issues = []
    
    if not relevant_chunk_ids:
        return issues
    
    # Собираем контекст из релевантных чанков
    context = "\n---\n".join([
        text_chunks[chunk_id]
        for chunk_id in relevant_chunk_ids
        if chunk_id < len(text_chunks)
    ])
    
    # Формируем prompt для LLM
    verification_prompt = f"""Проверьте ответ на вопрос в контексте материала учебника.

ВОПРОС:
{question.question_text}

ОТВЕТ:
{question.answer}

КОНТЕКСТ ИЗ УЧЕБНИКА:
{context}

АНАЛИЗ:
1. Является ли ответ ФАКТИЧЕСКИ КОРРЕКТНЫМ согласно учебнику?
2. ПОЛЕН ли ответ или в нём не хватает информации?
3. Является ли ответ СЛИШКОМ СЛОЖНЫМ для уровня вопроса?
4. ЯСЕН ли ответ для понимания?

ОТВЕТЬТЕ В ФОРМАТЕ JSON:
{{
  "factual_correct": true/false,
  "complete": true/false,
  "appropriate_complexity": true/false,
  "clarity": true/false,
  "issues": ["проблема1", "проблема2"],
  "suggestions": ["рекомендация1"]
}}

Будьте критичны и честны."""
    
    # Получаем ответ от LLM
    try:
        response = llm.invoke(verification_prompt)
        response_text = response if isinstance(response, str) else response.content
        
        # Парсим JSON из ответа
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            # Значения по умолчанию, если JSON не найден
            analysis = {
                "factual_correct": True,
                "complete": True,
                "appropriate_complexity": True,
                "clarity": True,
                "issues": [],
                "suggestions": []
            }
    except Exception:
        # На случай ошибки LLM
        analysis = {
            "factual_correct": True,
            "complete": True,
            "appropriate_complexity": True,
            "clarity": True,
            "issues": [],
            "suggestions": []
        }
    
    # Преобразуем анализ в VerificationIssue
    if not analysis.get("factual_correct", True):
        issues.append(VerificationIssue(
            type="factual_error",
            severity="high",
            description="Обнаружены фактические ошибки в ответе",
            suggested_fix=analysis.get("suggestions", [""])[0]
        ))
    
    if not analysis.get("complete", True):
        issues.append(VerificationIssue(
            type="incomplete_answer",
            severity="medium",
            description="Ответ неполный, не хватает информации",
            suggested_fix=analysis.get("suggestions", [""])[0]
        ))
    
    if not analysis.get("appropriate_complexity", True):
        issues.append(VerificationIssue(
            type="too_complex",
            severity="medium",
            description="Ответ слишком сложен или не соответствует уровню вопроса",
            suggested_fix=analysis.get("suggestions", [""])[0]
        ))
    
    return issues


def check_duplication_with_approved(
    question: Question,
    question_embedding: list[float],
    approved_questions: list[VerifiedQuestion],
    approved_embeddings: dict[int, list[float]]
) -> list[VerificationIssue]:
    """
    ПРОВЕРКА 3: Дублирование с одобренными вопросами
    """
    issues = []
    
    duplicates = check_duplication(
        question_embedding,
        approved_embeddings,
        duplication_threshold=0.9
    )
    
    if duplicates:
        issues.append(VerificationIssue(
            type="duplicated_question",
            severity="medium",
            description=f"Вопрос дублирует вопросы: {duplicates}",
            suggested_fix="Переформулируйте вопрос или задайте другой аспект темы"
        ))
    
    return issues


def determine_verification_status(
    relevant: bool,
    issues: list[VerificationIssue]
) -> tuple[Literal["approved", "needs_revision", "rejected"], str]:
    """
    ПРОВЕРКА 4: Определение финального статуса
    """
    if not relevant:
        return "rejected", "Вопрос не релевантен исходному материалу учебника"
    
    if not issues:
        return "approved", ""
    
    # Проверяем наличие HIGH-severity issues
    high_severity_issues = [i for i in issues if i.severity == "high"]
    
    if high_severity_issues:
        feedback = "Обнаружены критичные ошибки:\n"
        for issue in high_severity_issues:
            feedback += f"- {issue.description}\n  Рекомендация: {issue.suggested_fix}\n"
        return "rejected", feedback
    
    # Есть низкие/средние проблемы → требуется пересмотр
    feedback = "Требуется пересмотр:\n"
    for issue in issues:
        feedback += f"- {issue.description} ({issue.severity})\n  Рекомендация: {issue.suggested_fix}\n"
    
    return "needs_revision", feedback


def verify_single_question(
    question: Question,
    text_chunks: list[str],
    chunk_embeddings: list[list[float]],
    approved_questions: list[VerifiedQuestion],
    approved_embeddings: dict[int, list[float]],
    llm,
    embeddings_model,
    threshold_similarity: float = 0.6
) -> VerifiedQuestion:
    """
    Верифицирует ОДИН вопрос
    
    Выполняет все 4 проверки и возвращает VerifiedQuestion
    """
    # ПРОВЕРКА 1: Релевантность
    relevant_chunks, is_relevant, question_embedding = check_relevance(
        question, text_chunks, chunk_embeddings, embeddings_model, threshold_similarity
    )
    
    # ПРОВЕРКА 2: Корректность ответа
    answer_issues = check_answer_correctness(
        question, text_chunks, relevant_chunks, llm
    )
    
    # ПРОВЕРКА 3: Дублирование
    duplication_issues = check_duplication_with_approved(
        question, question_embedding, approved_questions, approved_embeddings
    )
    
    # Объединяем все проблемы
    all_issues = answer_issues + duplication_issues
    
    # ПРОВЕРКА 4: Статус
    status, feedback = determine_verification_status(is_relevant, all_issues)
    
    # Создаём результат
    verified_q = VerifiedQuestion(
        id=question.id,
        question_text=question.question_text,
        answer=question.answer,
        verification_status=status,
        issues=all_issues,
        source_chunks_ids=relevant_chunks,
        feedback=feedback
    )
    
    return verified_q, question_embedding


# ============================================================================
# 4. ГЛАВНАЯ ФУНКЦИЯ ВЕРИФИКАЦИИ
# ============================================================================

def verify_questions(
    questions: list[Question],
    text_chunks: list[str],
    chunk_embeddings: list[list[float]],
    llm,
    embeddings_model,
    threshold_similarity: float = 0.6
) -> VerifierOutput:
    """
    ГЛАВНАЯ ФУНКЦИЯ: Верифицирует все вопросы
    
    Args:
        questions: Список вопросов для проверки
        text_chunks: Чанки текста из учебника
        chunk_embeddings: Эмбединги чанков
        llm: LLM модель для анализа
        embeddings_model: Модель для вычисления эмбедингов
        threshold_similarity: Порог релевантности
    
    Returns:
        VerifierOutput с результатами верификации
    """
    try:
        # Валидация входных данных
        if not questions:
            raise ValueError("Список вопросов не может быть пустым")
        if not text_chunks or not chunk_embeddings:
            raise ValueError("Требуются текстовые чанки и эмбединги")
        if len(text_chunks) != len(chunk_embeddings):
            raise ValueError("Количество чанков должно совпадать с количеством эмбедингов")
        
        verified_questions: list[VerifiedQuestion] = []
        approved_questions: list[VerifiedQuestion] = []
        question_embeddings: dict[int, list[float]] = {}
        
        # Верифицируем каждый вопрос
        for question in questions:
            verified_q, q_embedding = verify_single_question(
                question,
                text_chunks,
                chunk_embeddings,
                approved_questions,
                {q.id: question_embeddings.get(q.id) for q in approved_questions 
                 if q.id in question_embeddings},
                llm,
                embeddings_model,
                threshold_similarity
            )
            
            verified_questions.append(verified_q)
            question_embeddings[question.id] = q_embedding
            
            if verified_q.verification_status == "approved":
                approved_questions.append(verified_q)
        
        # Готовим резюме
        total = len(verified_questions)
        approved = len([q for q in verified_questions if q.verification_status == "approved"])
        needs_revision = len([q for q in verified_questions if q.verification_status == "needs_revision"])
        rejected = len([q for q in verified_questions if q.verification_status == "rejected"])
        
        avg_score = approved / total if total > 0 else 0
        
        summary = {
            "total_verified": total,
            "approved_count": approved,
            "needs_revision_count": needs_revision,
            "rejected_count": rejected,
            "avg_verification_score": round(avg_score, 3),
            "approval_rate_percentage": round(100 * avg_score, 1)
        }
        
        return VerifierOutput(
            status="success",
            verified_questions=verified_questions,
            summary=summary
        )
    
    except Exception as e:
        return VerifierOutput(
            status="error",
            verified_questions=[],
            summary={},
            error_message=str(e)
        )