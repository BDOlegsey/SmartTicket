"""
SmartTicket Main Function - Загрузка PDF и запуск системы
Версия: 2.0
Дата: 2025-12-16

Главная функция для работы системы SmartTicket с PDF файлами
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Для работы с PDF
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 не установлен. Используйте: pip install PyPDF2")

from agents.orchestrator_main import SmartTicketOrchestrator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С PDF
# ============================================================================

def load_pdf_text(pdf_path: str) -> Optional[str]:
    """
    Загружает текст из PDF файла
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Текст из PDF или None если ошибка
    """
    if not PDF_AVAILABLE:
        logger.error("❌ PyPDF2 не установлен")
        return None
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        logger.error(f"❌ Файл не найден: {pdf_path}")
        return None
    
    try:
        logger.info(f"📖 Загрузка PDF: {pdf_path}")
        
        text = ""
        with open(pdf_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            
            logger.info(f"   Всего страниц: {num_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                if page_num % 10 == 0:
                    logger.info(f"   Обработано страниц: {page_num}/{num_pages}")
        
        logger.info(f"✓ PDF загружен успешно")
        logger.info(f"  Всего символов: {len(text)}")
        logger.info(f"  Всего слов (примерно): {len(text.split())}")
        
        return text
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке PDF: {e}")
        return None


def validate_text(text: str, min_length: int = 500) -> bool:
    """
    Проверяет качество загруженного текста
    
    Args:
        text: Текст для проверки
        min_length: Минимальная длина текста
        
    Returns:
        True если текст валиден
    """
    if not text:
        logger.error("❌ Текст пуст")
        return False
    
    if len(text) < min_length:
        logger.error(f"❌ Текст слишком короткий ({len(text)} символов, нужно >= {min_length})")
        return False
    
    # Проверить, есть ли хотя бы несколько слов
    words = text.split()
    if len(words) < 50:
        logger.error(f"❌ Слишком мало слов ({len(words)}, нужно >= 50)")
        return False
    
    logger.info(f"✓ Текст валиден:")
    logger.info(f"  Символы: {len(text)}")
    logger.info(f"  Слова: {len(words)}")
    logger.info(f"  Предложения: {len(text.split('.'))}")
    
    return True


def save_results_to_file(results: dict, output_dir: str = ".") -> str:
    """
    Сохраняет результаты в JSON файл
    
    Args:
        results: Результаты тестирования
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"smartticket_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Результаты сохранены в: {filename}")
        return str(filename)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении результатов: {e}")
        return ""


def print_results_summary(results: dict):
    """
    Выводит красивое резюме результатов
    
    Args:
        results: Результаты тестирования
    """
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ SmartTicket СИСТЕМА")
    print("=" * 80)
    
    if "error" in results:
        print(f"\n❌ ОШИБКА: {results['error']}")
        return
    
    print(f"\nВремя: {results.get('timestamp', 'N/A')}")
    
    # ANALYZER
    print(f"\n📈 ANALYZER AGENT:")
    analyzer = results.get('analyzer', {})
    print(f"  Success Rate:       {analyzer.get('success_rate', 0):.1f}%")
    print(f"  Embedding Quality:  {analyzer.get('embedding_quality', 0):.3f}")
    print(f"  Text Completeness:  {analyzer.get('text_completeness', 0):.1f}%")
    print(f"  Chunks Count:       {analyzer.get('chunks_count', 0)}")
    print(f"  Avg Chunk Size:     {analyzer.get('avg_chunk_size', 0):.0f} символов")
    status = "✓ PASS" if analyzer.get('success_rate', 0) >= 95 else "✗ FAIL"
    print(f"  Status:             {status}")
    
    # GENERATOR
    print(f"\n🎯 GENERATOR AGENT:")
    generator = results.get('generator', {})
    print(f"  Success Rate:       {generator.get('success_rate', 0):.1f}%")
    print(f"  Diversity Score:    {generator.get('diversity_score', 0):.3f}")
    print(f"  Complexity Index:   {generator.get('complexity_index', 0):.3f}")
    print(f"  Avg Question Len:   {generator.get('avg_question_length', 0):.0f}")
    print(f"  Avg Answer Len:     {generator.get('avg_answer_length', 0):.0f}")
    print(f"  Questions Generated:{generator.get('questions_generated', 0)}")
    status = "✓ PASS" if generator.get('diversity_score', 0) >= 0.4 else "✗ FAIL"
    print(f"  Status:             {status}")
    
    # CHECKER
    print(f"\n✓ CHECKER AGENT:")
    checker = results.get('checker', {})
    print(f"  Validation Ratio:   {checker.get('validation_ratio', 0):.1f}%")
    print(f"  Precision Score:    {checker.get('precision_score', 0):.1f}%")
    print(f"  Consistency Score:  {checker.get('consistency_score', 0):.3f}")
    print(f"  Issues Found:       {checker.get('issues_found', 0)}")
    status = "✓ PASS" if checker.get('consistency_score', 0) >= 0.6 else "✗ FAIL"
    print(f"  Status:             {status}")
    
    # SCANNER
    print(f"\n🔍 SCANNER AGENT:")
    scanner = results.get('scanner', {})
    print(f"  Performance:        {scanner.get('performance', 0):.3f}")
    print(f"  Detection Rate:     {scanner.get('detection_rate', 0):.1f}%")
    print(f"  Source Quality:     {scanner.get('source_quality', 0):.3f}")
    print(f"  Avg Similarity:     {scanner.get('avg_similarity', 0):.3f}")
    print(f"  Sources Count:      {scanner.get('sources_count', 0)}")
    status = "✓ PASS" if scanner.get('sources_count', 0) >= 3 else "✗ FAIL"
    print(f"  Status:             {status}")
    
    # SYSTEM
    print(f"\n⚙️  SYSTEM METRICS:")
    system = results.get('system', {})
    overall = system.get('overall_quality', 0)
    print(f"  Overall Quality:    {overall:.3f} (целевое: ≥0.75)")
    print(f"  E2E Time:           {system.get('e2e_time', 0):.3f}s (целевое: ≤2s)")
    print(f"  Stability:          {system.get('stability', 0):.1f}% (целевое: ≥95%)")
    
    # Финальный статус
    print("\n" + "=" * 80)
    if overall >= 0.75:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ - СИСТЕМА ГОТОВА К PRODUCTION")
    else:
        print(f"⚠️  СИСТЕМА ТРЕБУЕТ УЛУЧШЕНИЙ (Quality: {overall:.3f})")
    print("=" * 80 + "\n")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ MAIN
# ============================================================================

async def main(
    pdf_path: str = "input/textbook.pdf",
    num_questions: int = 10,
    output_dir: str = "./results",
    loop_iterations: int = 1
):
    """
    Главная функция для работы системы SmartTicket с PDF
    
    Процесс:
    1. Загружает текст из PDF
    2. Проверяет качество текста
    3. Запускает полный pipeline системы
    4. Сохраняет результаты
    5. Выводит итоговый отчет
    6. (Опционально) Повторяет процесс N раз
    
    Args:
        pdf_path: Путь к PDF файлу (default: "textbook.pdf")
        num_questions: Количество вопросов для генерации (default: 10)
        output_dir: Директория для сохранения результатов (default: "./results")
        loop_iterations: Количество итераций (default: 1)
    """
    
    print("\n" + "=" * 80)
    print("🚀 SMARTTICKET СИСТЕМА - ЗАПУСК")
    print("=" * 80 + "\n")
    
    # =========================================================================
    # ШАГ 1: ЗАГРУЗКА PDF
    # =========================================================================
    logger.info("\nШАГ 1: ЗАГРУЗКА И ВАЛИДАЦИЯ ТЕКСТА")
    logger.info("-" * 80)
    
    text = load_pdf_text(pdf_path)
    
    if not text or not validate_text(text):
        logger.error("❌ Не удалось загрузить или валидировать текст")
        return None
    
    all_results = []
    
    # =========================================================================
    # ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ
    # =========================================================================
    for iteration in range(loop_iterations):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ИТЕРАЦИЯ {iteration + 1}/{loop_iterations}")
        logger.info(f"{'=' * 80}")
        
        # Создать оркестратор для каждой итерации
        orchestrator = SmartTicketOrchestrator()
        
        # =====================================================================
        # ШАГ 2: ЗАПУСТИТЬ ПОЛНЫЙ PIPELINE
        # =====================================================================
        logger.info(f"\nШАГ 2: ЗАПУСК ПОЛНОГО PIPELINE")
        logger.info("-" * 80)
        
        try:
            results = await orchestrator.run_full_pipeline(text, num_questions)
            
            if results and "status" in results and results["status"] == "success":
                logger.info("\n✅ Pipeline завершен успешно")
                all_results.append(results)
            else:
                logger.error("❌ Pipeline завершился с ошибкой")
                if "error" in results:
                    logger.error(f"   Ошибка: {results['error']}")
                continue
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при запуске pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # =====================================================================
        # ШАГ 3: СОХРАНИТЬ РЕЗУЛЬТАТЫ
        # =====================================================================
        logger.info(f"\nШАГ 3: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        logger.info("-" * 80)
        
        output_file = save_results_to_file(results, output_dir)
        
        # =====================================================================
        # ШАГ 4: ВЫВЕСТИ ИТОГОВЫЙ ОТЧЕТ
        # =====================================================================
        logger.info(f"\nШАГ 4: ИТОГОВЫЙ ОТЧЕТ")
        logger.info("-" * 80)
        
        print_results_summary(results)
        
        # Пауза между итерациями если их несколько
        if iteration < loop_iterations - 1:
            logger.info(f"⏳ Пауза 5 секунд перед следующей итерацией...")
            await asyncio.sleep(5)
    
    # =========================================================================
    # ФИНАЛЬНОЕ РЕЗЮМЕ
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 ВСЕ ИТЕРАЦИИ ЗАВЕРШЕНЫ")
    logger.info("=" * 80 + "\n")
    
    if all_results:
        logger.info(f"Всего успешных запусков: {len(all_results)}")
        
        # Вычислить средние метрики
        if len(all_results) > 1:
            logger.info("\n📊 СРЕДНИЕ МЕТРИКИ ПО ВСЕМ ИТЕРАЦИЯМ:")
            
            avg_overall = sum(r.get("system", {}).get("overall_quality", 0) 
                            for r in all_results) / len(all_results)
            avg_quality = sum(r.get("analyzer", {}).get("embedding_quality", 0) 
                            for r in all_results) / len(all_results)
            avg_diversity = sum(r.get("generator", {}).get("diversity_score", 0) 
                              for r in all_results) / len(all_results)
            
            logger.info(f"  Average Overall Quality: {avg_overall:.3f}")
            logger.info(f"  Average Embedding Quality: {avg_quality:.3f}")
            logger.info(f"  Average Diversity Score: {avg_diversity:.3f}")
        
        logger.info(f"\n✅ Результаты сохранены в директорию: {output_dir}")
        return all_results
    else:
        logger.error("❌ Не было успешных запусков")
        return None


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    """
    Запуск с параметрами командной строки
    
    Примеры:
        python test_for_metrics.py
        python test_for_metrics.py --pdf textbook.pdf --questions 10
        python test_for_metrics.py --pdf textbook.pdf --questions 15 --iterations 3
        python test_for_metrics.py --pdf textbook.pdf --output ./my_results
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SmartTicket - Система автоматической генерации экзаменационных вопросов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python test_for_metrics.py
  python test_for_metrics.py --pdf textbook.pdf --questions 15
  python test_for_metrics.py --pdf textbook.pdf --iterations 3 --output ./results
  python test_for_metrics.py --help
        """
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        default="textbook.pdf",
        help="Путь к PDF файлу учебника (default: textbook.pdf)"
    )
    
    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        help="Количество вопросов для генерации (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Директория для сохранения результатов (default: ./results)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Количество итераций запуска (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Запустить main
    try:
        results = asyncio.run(main(
            pdf_path=args.pdf,
            num_questions=args.questions,
            output_dir=args.output,
            loop_iterations=args.iterations
        ))
        
        if results:
            sys.exit(0)  # Успех
        else:
            sys.exit(1)  # Ошибка
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Прервано пользователем")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
