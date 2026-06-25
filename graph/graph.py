from dotenv import load_dotenv

from langgraph.graph import END, START, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState


load_dotenv()


def _is_positive(value) -> bool:
    return value is True or str(value).strip().lower() in {"yes", "true", "1"}


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    if _is_positive(hallucination_score.binary_score):
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        answer_score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )

        if _is_positive(answer_score.binary_score):
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"

    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")

    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH

    if source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

    raise ValueError(f"Unknown datasource: {source.datasource}")


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")

    if _is_positive(state.get("web_search", False)):
        print("---DECISION: INCLUDE WEB SEARCH---")
        return WEBSEARCH

    print("---DECISION: GENERATE---")
    return GENERATE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()


if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")