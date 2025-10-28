# agents/generator.py
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL


def generate_tickets(state: dict) -> dict:
    """
    Генерирует экзаменационные билеты на основе учебника с использованием Qwen + RAG.
    """
    # Инициализация LLM
    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0.3,
        max_retries=2,
        timeout=60
    )

    # Получаем текст учебника (ограничим для промпта, если очень длинный)
    textbook = state["textbook_text"]
    # Для промпта возьмём первые 10 000 символов (можно адаптировать под контекстное окно)
    textbook_excerpt = textbook[:10000] if len(textbook) > 10000 else textbook

    # Формируем промпт
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Ты — опытный преподаватель. Твоя задача — создать экзаменационные билеты строго по предоставленному фрагменту учебника."),
        ("user", """
Учебник (фрагмент):
{excerpt}

Создай 3 экзаменационных билета по этому материалу.
Каждый билет должен содержать:
- 1 теоретический вопрос
- 1 прикладной или аналитический вопрос

Ответы на вопросы должны быть:
- Полными
- Основанными ТОЛЬКО на приведённом фрагменте учебника
- Чётко структурированными

Верни результат в формате JSON без дополнительного текста:
{{
  "tickets": [
    {{
      "questions": ["Вопрос 1", "Вопрос 2"],
      "answers": ["Полный ответ на вопрос 1...", "Полный ответ на вопрос 2..."]
    }},
    ...
  ]
}}
""")
    ])

    # Вызов модели
    chain = prompt_template | llm
    try:
        response = chain.invoke({"excerpt": textbook_excerpt})
        raw_output = response.content.strip()

        # Очистка от markdown-блоков, если есть
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        # Парсинг JSON
        parsed = json.loads(raw_output)

        # Преобразуем в формат, ожидаемый остальной системой
        tickets = []
        for t in parsed.get("tickets", []):
            tickets.append({
                "questions": t["questions"],
                "answers": t["answers"],
                "verified": False,
                "verification_iterations": 0
            })

        state["tickets"] = tickets
        state["draft_tickets"] = tickets

    except (json.JSONDecodeError, KeyError) as e:
        # Резервный вариант: логируем ошибку и создаём пустой билет для продолжения workflow
        print(f"⚠️ Ошибка генерации билетов: {e}")
        print(f"Сырой ответ модели:\n{raw_output}")
        state["tickets"] = [{
            "questions": ["Не удалось сгенерировать вопрос. Проверьте учебник и API."],
            "answers": ["Ошибка генерации."],
            "verified": False,
            "verification_iterations": 0
        }]

    return state