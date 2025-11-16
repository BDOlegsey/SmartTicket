# Ticket Checker Agent
# Агент для проверки и исправления сгенерированных билетов

import json
import openai


# Конфигурация OpenAI
def setup_openai_client(api_key: str) -> openai.OpenAI:
    """Инициализация OpenAI клиента"""
    return openai.OpenAI(api_key=api_key)


def validate_ticket(ticket: dict, prompt: str, text: str, client: openai.OpenAI, max_retries: int = 3) -> tuple[bool, str]:
    """
    Валидирует один билет на соответствие промпту и тексту
    
    Args:
        ticket: Сгенерированный билет
        prompt: Промпт, по которому был сгенерирован билет
        text: Исходный текст
        client: OpenAI клиент
        max_retries: Максимальное количество попыток валидации
    
    Returns:
        Кортеж (валиден ли, сообщение об ошибке или причина валидации)
    """
    
    validation_prompt = f"""
    Проверь, соответствует ли следующий билет (ticket) требованиям промпта и исходному тексту.
    
    Исходный текст:
    {text}
    
    Промпт для генерации:
    {prompt}
    
    Билет для проверки:
    {json.dumps(ticket, ensure_ascii=False, indent=2)}
    
    Ответь ТОЛЬКО в формате JSON:
    {{
        "is_valid": true/false,
        "reason": "краткое объяснение причины валидации или ошибки"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": validation_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["is_valid"], result["reason"]
    
    except json.JSONDecodeError:
        return False, "Ошибка парсинга ответа от OpenAI"
    except Exception as e:
        return False, f"Ошибка при валидации: {str(e)}"


def fix_ticket(ticket: dict, prompt: str, text: str, error_reason: str, client: openai.OpenAI) -> dict:
    """
    Исправляет невалидный билет
    
    Args:
        ticket: Невалидный билет
        prompt: Промпт для генерации
        text: Исходный текст
        error_reason: Причина ошибки валидации
        client: OpenAI клиент
    
    Returns:
        Исправленный билет
    """
    
    fix_prompt = f"""
    Исправь билет на основе ошибки валидации.
    
    Исходный текст:
    {text}
    
    Требования (промпт):
    {prompt}
    
    Текущий (невалидный) билет:
    {json.dumps(ticket, ensure_ascii=False, indent=2)}
    
    Причина ошибки:
    {error_reason}
    
    Исправь билет так, чтобы он полностью соответствовал требованиям и исходному тексту.
    Ответь ТОЛЬКО валидным JSON объектом билета, без дополнительного текста.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        fixed_ticket = json.loads(response.choices[0].message.content)
        return fixed_ticket
    
    except json.JSONDecodeError:
        print(f"Ошибка парсинга исправленного билета, возвращаем оригинальный")
        return ticket
    except Exception as e:
        print(f"Ошибка при исправлении билета: {str(e)}")
        return ticket


def process_tickets(text: str, prompt: str, tickets: list[dict], client: openai.OpenAI, max_iterations: int = 3) -> tuple[list[dict], list[tuple]]:
    """
    Обрабатывает список билетов, валидирует и циклично исправляет их
    
    Args:
        text: Исходный текст
        prompt: Промпт для генерации
        tickets: Список сгенерированных билетов
        client: OpenAI клиент
        max_iterations: Максимальное количество итераций исправления для каждого билета
    
    Returns:
        Список валидированных (и исправленных при необходимости) билетов; Список номеров обработанных билетов, которые исправить не удалось, и ошибки в них
    """
    
    processed_tickets = []
    bad_tickets = []
    
    for idx, ticket in enumerate(tickets, 1):
        print(f"Обработка билета #{idx}...")
        current_ticket = ticket
        iteration = 0
        
        while iteration <= max_iterations:
            # Валидируем билет
            is_valid, reason = validate_ticket(current_ticket, prompt, text, client)
            
            if is_valid:
                print(f"Билет #{idx} валиден!")
                print(f"Причина: {reason}")
                processed_tickets.append(current_ticket)
                break
            else:
                iteration += 1
                print(f"Билет #{idx} невалиден (попытка {iteration}/{max_iterations})")
                print(f"Ошибка: {reason}")
                
                if iteration <= max_iterations:
                    print(f"Исправляем билет...")
                    current_ticket = fix_ticket(current_ticket, prompt, text, reason, client)
                else:
                    print(f"Максимальное количество попыток исправления достигнуто")
                    processed_tickets.append(current_ticket)
                    bad_tickets.append((idx, reason))
        
    return processed_tickets, bad_tickets


def run_checker_agent(api_key: str, text: str, prompt: str, tickets: list[dict], max_iterations: int = 3) -> dict:
    """
    Главная функция агента-чекера
    
    Args:
        api_key: API ключ OpenAI
        text: Исходный текст
        prompt: Промпт для генерации билетов
        tickets: Список сгенерированных билетов
        max_iterations: Максимальное количество итераций исправления
    
    Returns:
        Словарь с результатами обработки
    """
    
    print("Запуск агента-чекера билетов...")
    print(f"Входные данные:")
    print(f"- Количество билетов: {len(tickets)}")
    print(f"- Максимум итераций: {max_iterations}")
    
    # Инициализируем клиент
    client = setup_openai_client(api_key)
    
    # Обрабатываем билеты
    processed_tickets, bad_tickets = process_tickets(text, prompt, tickets, client, max_iterations)
    
    # Компилируем результаты
    results = {
        "total_tickets": len(tickets),
        "processed_tickets": len(processed_tickets),
        "tickets": processed_tickets,
        "bad_tickets": bad_tickets,
        "status": "completed"
    }
    
    print(f"\n Обработка завершена!")
    print(f"Статистика: {results['processed_tickets']}/{results['total_tickets']} билетов обработано, из которых не удалось исправить {len(results['bad_tickets'])}")
    
    return results

