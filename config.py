"""
config.py - Конфигурация для выбора LLM провайдера (OpenAI или Qwen3)

Поддерживает:
- OpenAI (gpt-4-turbo, gpt-3.5-turbo и т.д.)
- Qwen3 (OpenAI-совместимый API через foundation-models.api.cloud.ru)
"""

import os
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Загрузить .env файл
load_dotenv()

# ============================================================================
# LLM PROVIDER SELECTION
# ============================================================================

LLM_PROVIDER: Literal["openai", "qwen"] = os.getenv("LLM_PROVIDER", "qwen").lower()

# ============================================================================
# QWEN3 CONFIGURATION
# ============================================================================

QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://foundation-models.api.cloud.ru/v1"
)
QWEN_MODEL = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen3-Next-80B-A3B-Instruct"
)
QWEN_API_KEY = "OWFlMDMyOWUtMGNiNi00OTg1LTk3MzItZWQzMWU3NzBkMzhi.2fe434181e5f3cdbde257f4462d87d2a"
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.7"))

class LLMConfig(BaseSettings):
    QWEN_BASE_URL: str = "https://foundation-models.api.cloud.ru/v1"
    QWEN_MODEL: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    QWEN_API_KEY: str = QWEN_API_KEY
    LLM_TIMEOUT: int = 120

llm_config = LLMConfig()

# ============================================================================
# COMMON PARAMETERS
# ============================================================================

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Проверка конфигурации"""
    
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "❌ OPENAI_API_KEY не установлен в .env файле"
            )
        print(f"✓ OpenAI конфигурация валидна (модель: {OPENAI_MODEL})")
    
    elif LLM_PROVIDER == "qwen":
        if not QWEN_API_KEY:
            raise ValueError(
                "❌ QWEN_API_KEY не установлен в .env файле"
            )
        print(f"✓ Qwen3 конфигурация валидна (модель: {QWEN_MODEL})")
    
    else:
        raise ValueError(
            f"❌ Неизвестный провайдер: {LLM_PROVIDER}. "
            f"Используйте 'openai' или 'qwen'"
        )


# ============================================================================
# LLM FACTORY
# ============================================================================

def get_llm(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Фабрика для создания LLM экземпляра
    
    Args:
        provider: 'openai' или 'qwen' (если None, используется из конфига)
        temperature: Температура модели (если None, используется из конфига)
        max_tokens: Максимум токенов (если None, используется из конфига)
    
    Returns:
        ChatOpenAI экземпляр для выбранного провайдера
    
    Raises:
        ValueError: Если провайдер неизвестен или не сконфигурирован
    """
    
    # Использовать параметры конфига если не переданы
    _provider = provider or LLM_PROVIDER
    _temperature = temperature if temperature is not None else (
        OPENAI_TEMPERATURE if _provider == "openai" else QWEN_TEMPERATURE
    )
    _max_tokens = max_tokens or LLM_MAX_TOKENS
    
    # ========================================================================
    # QWEN3 PROVIDER
    # ========================================================================
    
    if _provider == "qwen":
        if not QWEN_API_KEY:
            raise ValueError(
                "❌ QWEN_API_KEY не установлен в .env файле\n"
                "Установите переменную окружения:\n"
                "  QWEN_API_KEY=your-key-here"
            )
        
        from langchain_openai import ChatOpenAI
        
        try:
            llm = ChatOpenAI(
                base_url=QWEN_BASE_URL,
                api_key=QWEN_API_KEY,
                model=QWEN_MODEL,
                temperature=_temperature,
                max_tokens=_max_tokens,
                timeout=LLM_TIMEOUT,
                # Дополнительные параметры для совместимости
                model_kwargs={
                    "timeout": LLM_TIMEOUT,
                    "top_p": 0.9,
                }
            )
            
            print(f"✓ Qwen3 LLM инициализирована успешно")
            print(f"  Base URL: {QWEN_BASE_URL}")
            print(f"  Model: {QWEN_MODEL}")
            print(f"  Temperature: {_temperature}")
            
            return llm
        
        except Exception as e:
            raise Exception(
                f"❌ Ошибка инициализации Qwen3: {str(e)}\n"
                f"Проверьте:\n"
                f"  - QWEN_API_KEY в .env\n"
                f"  - QWEN_BASE_URL доступен\n"
                f"  - Интернет соединение"
            )
    
    # ========================================================================
    # OPENAI PROVIDER
    # ========================================================================
    
    elif _provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "❌ OPENAI_API_KEY не установлен в .env файле\n"
                "Установите переменную окружения:\n"
                "  OPENAI_API_KEY=sk-..."
            )
        
        from langchain_openai import ChatOpenAI
        
        try:
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                temperature=_temperature,
                max_tokens=_max_tokens,
                timeout=LLM_TIMEOUT,
            )
            
            print(f"✓ OpenAI LLM инициализирована успешно")
            print(f"  Model: {OPENAI_MODEL}")
            print(f"  Temperature: {_temperature}")
            
            return llm
        
        except Exception as e:
            raise Exception(
                f"❌ Ошибка инициализации OpenAI: {str(e)}\n"
                f"Проверьте:\n"
                f"  - OPENAI_API_KEY в .env\n"
                f"  - Интернет соединение"
            )
    
    else:
        raise ValueError(
            f"❌ Неизвестный провайдер: {_provider}\n"
            f"Используйте 'openai' или 'qwen'"
        )


# ============================================================================
# EMBEDDINGS FACTORY
# ============================================================================

def get_embeddings(provider: Optional[str] = None):
    """
    Фабрика для создания embeddings модели
    
    Args:
        provider: 'openai' или 'qwen' (если None, используется из конфига)
    
    Returns:
        Embeddings экземпляр
    """
    
    _provider = provider or LLM_PROVIDER
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        if _provider == "qwen":
            # Используем Qwen embeddings если доступны
            return OpenAIEmbeddings(
                base_url=QWEN_BASE_URL,
                api_key=QWEN_API_KEY,
                model="Qwen/Qwen1.5-Text-Embedding-104B",
            )
        else:
            # OpenAI embeddings
            return OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model="text-embedding-3-small",
            )
    
    except Exception as e:
        print(f"⚠️ Embeddings недоступны: {e}")
        print("   Система будет использовать mock embeddings")
        return None


# ============================================================================
# CONFIGURATION STATUS
# ============================================================================

def print_config():
    """Вывести информацию о текущей конфигурации"""
    
    print("\n" + "="*70)
    print("LLM КОНФИГУРАЦИЯ")
    print("="*70)
    
    print(f"\n📌 Текущий провайдер: {LLM_PROVIDER.upper()}")
    
    if LLM_PROVIDER == "qwen":
        print(f"\n🔷 Qwen3 конфигурация:")
        print(f"   Base URL: {QWEN_BASE_URL}")
        print(f"   Model: {QWEN_MODEL}")
        print(f"   API Key: {'✓ установлен' if QWEN_API_KEY else '❌ НЕ установлен'}")
        print(f"   Temperature: {QWEN_TEMPERATURE}")
        print(f"   Max Tokens: {LLM_MAX_TOKENS}")
        print(f"   Timeout: {LLM_TIMEOUT}s")
    
    elif LLM_PROVIDER == "openai":
        print(f"\n🔵 OpenAI конфигурация:")
        print(f"   Model: {OPENAI_MODEL}")
        print(f"   API Key: {'✓ установлен' if OPENAI_API_KEY else '❌ НЕ установлен'}")
        print(f"   Temperature: {OPENAI_TEMPERATURE}")
        print(f"   Max Tokens: {LLM_MAX_TOKENS}")
        print(f"   Timeout: {LLM_TIMEOUT}s")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN - Тестирование конфигурации
# ============================================================================

if __name__ == "__main__":
    print_config()
    
    try:
        validate_config()
        llm = get_llm()
        print("✓ LLM успешно инициализирована!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
