%%writefile server_app.py

from fastapi import FastAPI
from langserve import add_routes

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS

from operator import itemgetter

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain’s Runnable interfaces",
)

llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')


docstore = FAISS.load_local("./docstore_index", lambda x:x, allow_dangerous_deserialization=True)

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
), ('user', '{input}')])

long_reorder = RunnableLambda(LongContextReorder().transform_documents)  ## GIVEN
context_getter = itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str
retrieval_chain = {'input' : (lambda x: x)} | RunnableAssign({'context' : context_getter})

generator_chain = RunnableAssign({"output" : chat_prompt | llm })
generator_chain = generator_chain | output_puller
rag_chain = retrieval_chain | generator_chain



add_routes(
    app,
    llm,
    path="/basic_chat",
)

add_routes(
    app,
    retrieval_chain,
    path="/retriever",
)

add_routes(
    app,
    generator_chain,
    path="/generator",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
