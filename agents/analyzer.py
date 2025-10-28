from utils.rag import RAGPipeline

def analyze_textbook(textbook_text: str) -> dict:
    rag = RAGPipeline(textbook_text)
    return {"textbook_text": textbook_text, "rag": rag}