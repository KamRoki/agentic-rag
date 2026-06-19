from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings


urls = [
    'https://www.mindstudio.ai/blog/andrej-karpathy-llm-wiki-knowledge-base-claude-code',
    'https://medium.com/@urvvil08/andrej-karpathys-llm-wiki-create-your-own-knowledge-base-8779014accd5',
    'https://datasciencedojo.com/blog/llm-wiki-tutorial/',
    'https://www.dume.ai/blog/what-is-andrej-karpathys-llm-wiki-how-to-get-the-same-results-without-code-using-dume-cowork'
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 250,
                                                                     chunk_overlap = 0)
doc_splits = text_splitter.split_documents(docs_list)


vectorstore = Chroma.from_documents(
    documents = doc_splits,
    collection_name = 'rag-chroma-data',
    embedding = OllamaEmbeddings(model = 'rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest'),
    persist_directory = './.chroma'
)


retriever = Chroma(
    collection_name = 'rag-chroma-data',
    persist_directory = './.chroma',
    embedding_function = OllamaEmbeddings(model = 'rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest')
).as_retriever()