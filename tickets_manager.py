"""
SmartTicket Tickets Manager - Сохранение и управление билетами
Версия: 2.1
Дата: 2025-12-16

Расширенный функционал для сохранения сгенерированных билетов в БД
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# МОДЕЛИ ДАННЫХ ДЛЯ БИЛЕТОВ
# ============================================================================

class TicketData:
    """Данные одного билета"""
    
    def __init__(self, ticket_id: int, questions: List[Dict], source_pdf: str, 
                 generation_date: str, overall_quality: float):
        self.ticket_id = ticket_id
        self.questions = questions  # Список вопросов с ответами
        self.source_pdf = source_pdf
        self.generation_date = generation_date
        self.overall_quality = overall_quality
    
    def to_dict(self) -> Dict:
        """Преобразовать в словарь"""
        return {
            "ticket_id": self.ticket_id,
            "questions": self.questions,
            "source_pdf": self.source_pdf,
            "generation_date": self.generation_date,
            "overall_quality": self.overall_quality,
            "question_count": len(self.questions)
        }


# ============================================================================
# DATABASE MANAGER - СОХРАНЕНИЕ БИЛЕТОВ В SQLite
# ============================================================================

class TicketsDatabaseManager:
    """Управление базой данных билетов"""
    
    def __init__(self, db_path: str = "./smartticket.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"✓ TicketsDatabaseManager инициализирован")
    
    def init_database(self):
        """Инициализирует БД если её нет"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица билетов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    generation_date TEXT NOT NULL,
                    overall_quality REAL NOT NULL,
                    question_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица вопросов (связь один-ко-многим)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id INTEGER NOT NULL,
                    question_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    source_chunk TEXT,
                    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ БД инициализирована: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при инициализации БД: {e}")
    
    def save_ticket(self, ticket: TicketData) -> bool:
        """Сохраняет билет в БД"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Сохранить основные данные билета
            cursor.execute("""
                INSERT INTO tickets (ticket_id, source_pdf, generation_date, 
                                   overall_quality, question_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticket.ticket_id,
                ticket.source_pdf,
                ticket.generation_date,
                ticket.overall_quality,
                len(ticket.questions)
            ))
            
            # Сохранить все вопросы
            for q in ticket.questions:
                cursor.execute("""
                    INSERT INTO questions (ticket_id, question_text, answer_text, 
                                         question_type, source_chunk)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    ticket.ticket_id,
                    q.get("question", ""),
                    q.get("answer", ""),
                    q.get("type", "unknown"),
                    q.get("source_chunk", "")
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Билет #{ticket.ticket_id} сохранен в БД")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении билета: {e}")
            return False
    
    def get_ticket(self, ticket_id: int) -> Optional[Dict]:
        """Получить билет по ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получить основные данные
            cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
            ticket_row = cursor.fetchone()
            
            if not ticket_row:
                return None
            
            # Получить вопросы
            cursor.execute("""
                SELECT question_text, answer_text, question_type, source_chunk 
                FROM questions WHERE ticket_id = ?
            """, (ticket_id,))
            questions = cursor.fetchall()
            
            conn.close()
            
            return {
                "ticket_id": ticket_row[0],
                "source_pdf": ticket_row[1],
                "generation_date": ticket_row[2],
                "overall_quality": ticket_row[3],
                "question_count": ticket_row[4],
                "questions": [
                    {
                        "question": q[0],
                        "answer": q[1],
                        "type": q[2],
                        "source_chunk": q[3]
                    }
                    for q in questions
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка при получении билета: {e}")
            return None
    
    def get_all_tickets(self, pdf_name: str = None) -> List[Dict]:
        """Получить все билеты или по конкретному PDF"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if pdf_name:
                cursor.execute("""
                    SELECT ticket_id, source_pdf, generation_date, overall_quality, 
                           question_count FROM tickets WHERE source_pdf = ?
                    ORDER BY creation_date DESC
                """, (pdf_name,))
            else:
                cursor.execute("""
                    SELECT ticket_id, source_pdf, generation_date, overall_quality, 
                           question_count FROM tickets
                    ORDER BY creation_date DESC
                """)
            
            tickets = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "ticket_id": t[0],
                    "source_pdf": t[1],
                    "generation_date": t[2],
                    "overall_quality": t[3],
                    "question_count": t[4]
                }
                for t in tickets
            ]
            
        except Exception as e:
            logger.error(f"❌ Ошибка при получении списка билетов: {e}")
            return []
    
    def delete_ticket(self, ticket_id: int) -> bool:
        """Удалить билет"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM questions WHERE ticket_id = ?", (ticket_id,))
            cursor.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Билет #{ticket_id} удален из БД")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении билета: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Получить статистику по билетам"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Всего билетов
            cursor.execute("SELECT COUNT(*) FROM tickets")
            total_tickets = cursor.fetchone()[0]
            
            # Всего вопросов
            cursor.execute("SELECT COUNT(*) FROM questions")
            total_questions = cursor.fetchone()[0]
            
            # Средняя оценка качества
            cursor.execute("SELECT AVG(overall_quality) FROM tickets")
            avg_quality = cursor.fetchone()[0]
            
            # Источники (PDF)
            cursor.execute("SELECT DISTINCT source_pdf FROM tickets")
            sources = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_tickets": total_tickets,
                "total_questions": total_questions,
                "average_quality": float(avg_quality) if avg_quality else 0.0,
                "sources_count": len(sources),
                "sources": [s[0] for s in sources]
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка при получении статистики: {e}")
            return {}


# ============================================================================
# JSON FILE MANAGER - СОХРАНЕНИЕ В JSON ФАЙЛЫ
# ============================================================================

class TicketsJSONManager:
    """Сохранение билетов в JSON файлы"""
    
    def __init__(self, output_dir: str = "./tickets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ TicketsJSONManager инициализирован")
    
    def save_ticket_json(self, ticket: TicketData) -> str:
        """Сохраняет билет в JSON файл"""
        try:
            filename = self.output_dir / f"ticket_{ticket.ticket_id:03d}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(ticket.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Билет #{ticket.ticket_id} сохранен в JSON: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении JSON: {e}")
            return ""
    
    def load_ticket_json(self, ticket_id: int) -> Optional[Dict]:
        """Загружает билет из JSON файла"""
        try:
            filename = self.output_dir / f"ticket_{ticket_id:03d}.json"
            
            if not filename.exists():
                logger.warning(f"⚠️ Файл билета не найден: {filename}")
                return None
            
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке JSON: {e}")
            return None
    
    def save_tickets_batch(self, tickets: List[TicketData]) -> List[str]:
        """Сохраняет пакет билетов"""
        filenames = []
        for ticket in tickets:
            filename = self.save_ticket_json(ticket)
            if filename:
                filenames.append(filename)
        
        logger.info(f"✓ Сохранено {len(filenames)} билетов")
        return filenames
    
    def export_tickets_catalog(self, tickets: List[TicketData]) -> str:
        """Экспортирует каталог всех билетов"""
        try:
            catalog = {
                "timestamp": datetime.now().isoformat(),
                "total_tickets": len(tickets),
                "tickets": [t.to_dict() for t in tickets]
            }
            
            filename = self.output_dir / "tickets_catalog.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Каталог билетов сохранен: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении каталога: {e}")
            return ""


# ============================================================================
# PDF EXPORT - ЭКСПОРТ БИЛЕТОВ В PDF
# ============================================================================

class TicketsPDFExporter:
    """Экспорт билетов в PDF (требует reportlab)"""
    
    def __init__(self, output_dir: str = "./tickets_pdf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
            
            self.has_reportlab = True
        except ImportError:
            self.has_reportlab = False
            logger.warning("⚠️ reportlab не установлен. PDF экспорт недоступен.")
            logger.warning("   Установите: pip install reportlab")
    
    def export_ticket_pdf(self, ticket: TicketData) -> str:
        """Экспортирует билет в PDF"""
        if not self.has_reportlab:
            logger.error("❌ reportlab не установлен")
            return ""
        
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table
            from reportlab.lib import colors
            
            filename = self.output_dir / f"ticket_{ticket.ticket_id:03d}.pdf"
            
            doc = SimpleDocTemplate(str(filename), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Заголовок
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f2124'),
                spaceAfter=12,
                alignment=1
            )
            story.append(Paragraph(f"БИЛЕТ № {ticket.ticket_id:03d}", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Информация о билете
            info = f"Источник: {ticket.source_pdf} | Дата: {ticket.generation_date} | Качество: {ticket.overall_quality:.2f}"
            story.append(Paragraph(info, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Вопросы
            for i, q in enumerate(ticket.questions, 1):
                question_style = ParagraphStyle(
                    'Question',
                    parent=styles['Heading3'],
                    fontSize=12,
                    textColor=colors.HexColor('#208099'),
                    spaceAfter=6
                )
                
                q_text = f"Вопрос {i}: {q.get('question', '')}"
                story.append(Paragraph(q_text, question_style))
                
                a_text = f"<b>Ответ:</b> {q.get('answer', '')}"
                story.append(Paragraph(a_text, styles['Normal']))
                
                story.append(Spacer(1, 0.2*inch))
                
                # Разделитель
                if i < len(ticket.questions):
                    story.append(Spacer(1, 0.1*inch))
            
            doc.build(story)
            
            logger.info(f"✓ Билет #{ticket.ticket_id} экспортирован в PDF: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Ошибка при экспорте в PDF: {e}")
            return ""


# ============================================================================
# MAIN TICKETS MANAGER - ГЛАВНЫЙ МЕНЕДЖЕР БИЛЕТОВ
# ============================================================================

class TicketsManager:
    """Главный менеджер для управления билетами"""
    
    def __init__(self, db_path: str = "./smartticket.db", output_dir: str = "./tickets"):
        self.db_manager = TicketsDatabaseManager(db_path)
        self.json_manager = TicketsJSONManager(output_dir)
        self.pdf_exporter = TicketsPDFExporter(f"{output_dir}/pdf")
        
        self.ticket_counter = 0
        logger.info("✓ TicketsManager инициализирован")
    
    def save_ticket(self, questions: List[Dict], source_pdf: str, 
                   overall_quality: float, export_formats: List[str] = None) -> int:
        """
        Сохраняет билет во все форматы
        
        Args:
            questions: Список вопросов с ответами
            source_pdf: Имя исходного PDF файла
            overall_quality: Общее качество билета
            export_formats: Форматы для экспорта ['db', 'json', 'pdf']
        
        Returns:
            ID сохраненного билета
        """
        if export_formats is None:
            export_formats = ['db', 'json']
        
        self.ticket_counter += 1
        ticket_id = self.ticket_counter
        
        # Создать объект билета
        ticket = TicketData(
            ticket_id=ticket_id,
            questions=questions,
            source_pdf=source_pdf,
            generation_date=datetime.now().isoformat(),
            overall_quality=overall_quality
        )
        
        # Сохранить в БД
        if 'db' in export_formats:
            self.db_manager.save_ticket(ticket)
        
        # Сохранить в JSON
        if 'json' in export_formats:
            self.json_manager.save_ticket_json(ticket)
        
        # Сохранить в PDF
        if 'pdf' in export_formats:
            self.pdf_exporter.export_ticket_pdf(ticket)
        
        logger.info(f"✓ Билет #{ticket_id} сохранен во все форматы")
        
        return ticket_id
    
    def get_ticket(self, ticket_id: int) -> Optional[Dict]:
        """Получить билет"""
        return self.db_manager.get_ticket(ticket_id)
    
    def get_all_tickets(self) -> List[Dict]:
        """Получить все билеты"""
        return self.db_manager.get_all_tickets()
    
    def get_statistics(self) -> Dict:
        """Получить статистику"""
        return self.db_manager.get_statistics()
    
    def export_tickets_catalog(self) -> str:
        """Экспортировать каталог"""
        all_tickets = [
            TicketData(
                ticket_id=t['ticket_id'],
                questions=[],
                source_pdf=t['source_pdf'],
                generation_date=t['generation_date'],
                overall_quality=t['overall_quality']
            )
            for t in self.db_manager.get_all_tickets()
        ]
        
        return self.json_manager.export_tickets_catalog(all_tickets)


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Создать менеджер билетов
    tickets_manager = TicketsManager()
    
    # Пример вопросов
    sample_questions = [
        {
            "question": "Что такое машинное обучение?",
            "answer": "Машинное обучение - это раздел искусственного интеллекта...",
            "type": "definition",
            "source_chunk": "Раздел об искусственном интеллекте..."
        },
        {
            "question": "Как работают нейронные сети?",
            "answer": "Нейронные сети моделируют работу мозга...",
            "type": "explain",
            "source_chunk": "Раздел о нейронных сетях..."
        }
    ]
    
    # Сохранить билет
    ticket_id = tickets_manager.save_ticket(
        questions=sample_questions,
        source_pdf="test_textbook.pdf",
        overall_quality=0.81,
        export_formats=['db', 'json', 'pdf']
    )
    
    print(f"\n✓ Билет #{ticket_id} сохранен")
    
    # Получить билет
    ticket = tickets_manager.get_ticket(ticket_id)
    print(f"\nБилет из БД:")
    print(json.dumps(ticket, indent=2, ensure_ascii=False))
    
    # Статистика
    stats = tickets_manager.get_statistics()
    print(f"\nСтатистика:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
