from dotenv import load_dotenv

load_dotenv()

from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from graph.chains.router import question_router, RouteQuery


def test_retrieval_grader_answer_yes() -> None:
    question = "knowledge base"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {
            "question": question,
            "documents": doc_txt
        }
    )
    
    assert res.binary_score == "yes"
    
    
def test_retrieval_grader_answer_no() -> None:
    question = "knowledge base"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {
            "question": "How to make pizza?",
            "documents": doc_txt
        }
    )
    
    assert res.binary_score == "no"
    
    
def test_generation_chain() -> None:
    question = "knowledge base"
    docs = retriever.invoke(question) 
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
    
    
def test_hallucination_grader_answer_yes() -> None:
    question = "knowledge base"
    docs = retriever.invoke(question) 
    
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke({"documents": docs, "generation": generation})
    
    assert res.binary_score
    
    
def test_hallucination_grader_answer_no() -> None:
    question = "knowledge base"
    docs = retriever.invoke(question) 
    
    generation = "In order to make pizza we need to first start with the dough."
    res: GradeHallucination = hallucination_grader.invoke({
        "documents": docs, 
        "generation": generation
    })
    
    assert not res.binary_score
    
    
def test_router_to_vectorstore() -> None:
    question = "knowledge base"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"
    
    
def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
    
    