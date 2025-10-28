from langchain_openai import ChatOpenAI
from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL


def verify_tickets(state: dict) -> dict:
    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0
    )

    textbook = state["textbook_text"]
    tickets = state.get("tickets", [])
    verified = []

    for t in tickets:
        q = t["questions"][0]
        a = t["answers"][0]
        prompt = f"""
        Учебник: {textbook[:1000]}
        Вопрос: {q}
        Ответ: {a}
        Проверь: 1) полнота, 2) точность по учебнику. Верни "OK" или "REJECT".
        """
        res = llm.invoke(prompt).content.strip()
        if "OK" in res:
            verified.append({**t, "verified": True, "verification_iterations": 1})
        else:
            # В реальной системе — возврат на генерацию
            verified.append({**t, "verified": False, "verification_iterations": 1})

    state["tickets"] = verified
    return state