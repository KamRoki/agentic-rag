from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


if __name__ == "__main__":
    print("AGENTIC RAG by Kamil Stachurski")
    
    print(app.invoke(input = {
        "question": "Co to jest koncept Wiki Andrej Karpathyiego?"
    }))