import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.analyzer import analyze_textbook
from agents.generator import generate_tickets
from agents.verifier import verify_tickets
from agents.web_scout import find_existing_tickets
from utils.pdf_loader import load_textbook


class GraphState(TypedDict):
    textbook_path: str
    textbook_text: str
    rag: object
    tickets: List[dict]
    draft_tickets: List[dict]
    external_examples: List[dict]


def load_textbook_node(state: GraphState) -> GraphState:
    text = load_textbook(state["textbook_path"])
    state["textbook_text"] = text
    return state


# Инициализация графа
workflow = StateGraph(GraphState)

workflow.add_node("load", load_textbook_node)
workflow.add_node("analyze", lambda s: analyze_textbook(s["textbook_text"]))
workflow.add_node("generate", generate_tickets)
workflow.add_node("verify", verify_tickets)
workflow.add_node("web_scout", find_existing_tickets)

workflow.set_entry_point("load")
workflow.add_edge("load", "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", "web_scout")
workflow.add_edge("web_scout", END)

app = workflow.compile()

# Запуск
if __name__ == "__main__":
    result = app.invoke({"textbook_path": "types.pdf"})

    output = {
        "source_textbook": "types.pdf",
        "generated_tickets": result["tickets"],
        "external_examples": result.get("external_examples", [])
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    input()


