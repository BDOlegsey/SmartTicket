"""
SHARED DATA MODELS - Унифицированные структуры данных для всех агентов
Эти модели используются для передачи данных между агентами
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


# ============================================================================
# PHASE 1: PDF анализ (AnalyzerAgent OUTPUT)
# ============================================================================

class TextChunk(BaseModel):
    """Фрагмент текста с метаданными"""
    chunk_id: int
    text: str
    page_number: Optional[int] = None
    source_file: str = ""


class AnalyzerOutput(BaseModel):
    """Выход AnalyzerAgent - анализированный PDF"""
    status: Literal["success", "error"]
    text_chunks: List[TextChunk]
    chunk_embeddings: List[List[float]]  # OpenAI embeddings (1536 dimensions для ada-002)
    total_chunks: int
    source_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "text_chunks": [{"chunk_id": 0, "text": "...", "page_number": 1}],
                "chunk_embeddings": [[0.1, 0.2, ...]],
                "total_chunks": 100,
                "source_file": "textbook.pdf"
            }
        }


# ============================================================================
# PHASE 2: Генерация билетов (GeneratorAgent INPUT/OUTPUT)
# ============================================================================

class Question(BaseModel):
    """Вопрос экзаменационного билета (УНИФИЦИРОВАННАЯ ВЕРСИЯ)"""
    id: int
    # Основное содержание
    question_text: str = Field(..., alias="question")  # Совместимость с GeneratorAgent
    answer_text: str = Field(..., alias="draft_answer")  # Совместимость с GeneratorAgent
    # Контекст из источника
    source_chunk_ids: List[int] = Field(default_factory=list)
    rag_context: List[str] = Field(default_factory=list)
    # Метрика качества
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True  # Для поддержки старых имен полей


class ExamTicket(BaseModel):
    """Экзаменационный билет"""
    ticket_id: str
    questions: List[Question]
    topic: str
    total_questions: int
    created_at: datetime = Field(default_factory=datetime.now)


class GeneratorOutput(BaseModel):
    """Выход GeneratorAgent"""
    status: Literal["success", "error"]
    tickets: List[ExamTicket]
    total_generated: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# ============================================================================
# PHASE 3: Проверка вопросов (CheckerAgent INPUT/OUTPUT)
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


class VerifiedQuestion(BaseModel):
    """Проверенный вопрос"""
    id: int
    question_text: str
    answer_text: str
    verification_status: Literal["approved", "needs_revision", "rejected"]
    issues: List[VerificationIssue] = Field(default_factory=list)
    source_chunks_ids: List[int] = Field(default_factory=list)
    feedback: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CheckerOutput(BaseModel):
    """Выход CheckerAgent"""
    status: Literal["success", "error"]
    verified_questions: List[VerifiedQuestion]
    summary: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# ============================================================================
# PHASE 4: Проверка плагиата (ScannerAgent INPUT/OUTPUT)
# ============================================================================

class ExternalSource(BaseModel):
    """Найденный внешний источник"""
    url: str
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    questions_found: int = 0
    snippet: str = ""
    source_authority: str = "low"
    domain: str = ""


class ScannerOutput(BaseModel):
    """Выход ScannerAgent - отчет о плагиате"""
    status: Literal["success", "partial", "not_found", "error"]
    found_plagiarism: bool
    sources: List[ExternalSource]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# ============================================================================
# FINAL: Итоговый отчет системы
# ============================================================================

class SystemReport(BaseModel):
    """Финальный отчет системы"""
    timestamp: datetime = Field(default_factory=datetime.now)
    source_file: str
    
    # Результаты каждого этапа
    analyzer_output: AnalyzerOutput
    generator_output: GeneratorOutput
    checker_output: CheckerOutput
    scanner_output: ScannerOutput
    
    # Итоговые метрики
    total_questions_generated: int
    total_questions_approved: int
    approval_rate: float
    average_confidence: float
    plagiarism_detected: bool
    
    # Статус системы
    overall_status: Literal["success", "partial_success", "error"]
    
    class Config:
        json_schema_extra = {
            "description": "Полный отчет системы SmartTicket о созданных экзаменационных билетах"
        }
