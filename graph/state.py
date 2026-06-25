from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Respresents the state of the graph.
    
    Attributes:
        - question: user question
        - generation: LLM generation
        - web_search: whether to add search
        - documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]