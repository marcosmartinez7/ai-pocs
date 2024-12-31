from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")


conversation = [  ## This conversation was generated partially by an AI system, and modified to exhibit desirable properties
    "[User]  Hello! My name is Beras, and I'm a big blue bear! Can you please tell me about the rocky mountains?",
    "[Agent] The Rocky Mountains are a beautiful and majestic range of mountains that stretch across North America",
    "[Beras] Wow, that sounds amazing! Ive never been to the Rocky Mountains before, but Ive heard many great things about them.",
    "[Agent] I hope you get to visit them someday, Beras! It would be a great adventure for you!"
    "[Beras] Thank you for the suggestion! Ill definitely keep it in mind for the future.",
    "[Agent] In the meantime, you can learn more about the Rocky Mountains by doing some research online or watching documentaries about them."
    "[Beras] I live in the arctic, so I'm not used to the warm climate there. I was just curious, ya know!",
    "[Agent] Absolutely! Lets continue the conversation and explore more about the Rocky Mountains and their significance!"
]

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS

## Streamlined from_texts FAISS vectorstore construction from text list
convstore = FAISS.from_texts(conversation, embedding=embedder)
retriever = convstore.as_retriever()

# Define the query
query = "What is your name?"

# Perform similarity search with scores
results_with_scores = convstore.similarity_search_with_score(query, k=4)  # k=4 retrieves top 4 similar documents

# Print the results with their similarity scores
for doc, score in results_with_scores:
    pprint(f"Score: {score:.4f}")
    pprint(doc.page_content)
    pprint("-" * 50)  # Separator for readability

pprint(retriever.invoke("Where are the Rocky Mountains?"))


from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from functools import partial
from operator import itemgetter

########################################################################
## Utility Runnables/Methods
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## Optional; Reorders longer documents to center of output text
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

context_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {question}"
    "\nAnswer the user conversationally. User is not aware of context."
)


chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'question': (lambda x:x)
    }
    | context_prompt
    | RPrint()
    | instruct_llm
    | StrOutputParser()
)

pprint(chain.invoke("Where does Beras live?"))
pprint(chain.invoke("How far away is Defensor from the Rocky Mountains?"))

convstore = FAISS.from_texts(conversation, embedding=embedder)

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([f"User said {d.get('input')}", f"Agent said {d.get('output')}"])
    return d.get('output')

########################################################################

# instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

chat_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {input}"
    "\nAnswer the user conversationally. Make sure the conversation flows naturally.\n"
    "[Agent]"
)


conv_chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'input': (lambda x:x)
    }
    | RPrint()
    | RunnableAssign({'output' : chat_prompt | instruct_llm | StrOutputParser()})
    | partial(save_memory_and_get_output, vstore=convstore)
)

pprint(conv_chain.invoke("I'm glad you agree! I can't wait to get some ice cream there! It's such a good food!"))
print()
pprint(conv_chain.invoke("Can you guess what my favorite food is?"))
print()
pprint(conv_chain.invoke("Actually, my favorite is honey! Not sure where you got that idea?"))
print()
pprint(conv_chain.invoke("I see! Fair enough! Do you know my favorite food now?"))

print("--"*65)

import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

## TODO: Please pick some papers and add them to the list as you'd like
## NOTE: To re-use for the final assessment, make sure at least one paper is < 1 month old
print("Loading Documents")
docs = [
    #ArxivLoader(query="1706.03762").load(),  ## Attention Is All You Need Paper
    #ArxivLoader(query="1810.04805").load(),  ## BERT Paper
    ArxivLoader(query="2005.11401").load(),  ## RAG Paper
    #ArxivLoader(query="2205.00445").load(),  ## MRKL Paper
    #ArxivLoader(query="2310.06825").load(),  ## Mistral Paper
    #ArxivLoader(query="2306.05685").load(),  ## LLM-as-a-Judge
    ## Some longer papers
    # ArxivLoader(query="2210.03629").load(),  ## ReAct Paper
    # ArxivLoader(query="2112.10752").load(),  ## Latent Stable Diffusion Paper
    # ArxivLoader(query="2103.00020").load(),  ## CLIP Paper
    ## TODO: Feel free to add more
]

## Cut the paper short if references is included.
## This is a standard string in papers.
for doc in docs:
    content = json.dumps(doc[0].page_content)
    if "References" in content:
        doc[0].page_content = content[:content.index("References")]

## Split the documents and also filter out stubs (overly short chunks)
print("Chunking Documents")
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

## Make some custom Chunks to give big-picture details
doc_string = "Available Documents:"
doc_metadata = []
for chunks in docs_chunks:
    metadata = getattr(chunks[0], 'metadata', {})
    doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

extra_chunks = [doc_string] + doc_metadata

## Printing out some summary information for reference
pprint(doc_string, '\n')
for i, chunks in enumerate(docs_chunks):
    print(f"Document {i}")
    print(f" - # Chunks: {len(chunks)}")
    print(f" - Metadata: ")
    pprint(chunks[0].metadata)
    print()

print("Constructing Vector Stores")
vecstores = [FAISS.from_texts(extra_chunks, embedder)]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
docstore = aggregate_vstores(vecstores)

print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## EXERCISE ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr
from functools import partial
from operator import itemgetter

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
)

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
), ('user', '{input}')])

stream_chain = chat_prompt| RPrint() | instruct_llm | StrOutputParser()

################################################################################################
## BEGIN TODO: Implement the retrieval chain to make your system work!

retrieval_chain = (
    {'input' : (lambda x: x)}
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    ## HINT: Our solution uses RunnableAssign, itemgetter, long_reorder, and docs2str
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)


## END TODO
################################################################################################

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ## First perform the retrieval based on the input message
    retrieval = retrieval_chain.invoke(message)
    line_buffer = ""

    ## Then, stream the results of the stream_chain
    for token in stream_chain.stream(retrieval):
        buffer += token
        ## If you're using standard print, keep line from getting too long
        yield buffer if return_buffer else token

    ## Lastly, save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)


## Start of Agent Event Loop
test_question = "Tell me about RAG!"  ## <- modify as desired

## Before you launch your gradio interface, make sure your thing works
for response in chat_gen(test_question, return_buffer=False):
    print(response, end='')

## Save and compress your index
docstore.save_local("docstore_index")


from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
new_db = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = new_db.similarity_search("Testing the index")
print(docs[0].page_content[:1000])