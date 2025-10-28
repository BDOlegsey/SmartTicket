from googlesearch import search


def find_existing_tickets(state: dict) -> dict:
    query = f"экзаменационные билеты по информатике учебник"
    results = []
    try:
        for url in search(query, num_results=3):
            results.append({
                "url": url,
                "description": "Найдены внешние билеты",
                "sample_questions": ["Пример вопроса"],
                "source": url
            })
    except Exception as e:
        print(f"Поиск не удался: {e}")

    state["external_examples"] = results
    return state