
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
norm_style = Style(bold=True)
pprint = partial(console.print, style=base_style)
pprint2 = partial(console.print, style=norm_style)

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models(base_url="http://llm_client:9000/v1")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

## Make sure you have docstore_index.tgz in your working directory
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4", truncate="END")

docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = list(docstore.docstore._dict.values())

def format_chunk(doc):
    return (
        f"Paper: {doc.metadata.get('Title', 'unknown')}"
        f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )

## This printout just confirms that your store has been retrieved
pprint(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
pprint(f"Sample Chunk:")
print(format_chunk(docs[len(docs)//2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.document_transformers import LongContextReorder

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from functools import partial
from operator import itemgetter

import gradio as gr

#####################################################################

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

#####################################################################

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

#####################################################################
## TODO: Pull in your desired RAG Chain. Memory not necessary

## Chain 1 Specs: "Hello World" -> retrieval_chain 
##   -> {'input': <str>, 'context' : <str>}
long_reorder = RunnableLambda(LongContextReorder().transform_documents)  ## GIVEN
context_getter = itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str
retrieval_chain = {'input' : (lambda x: x)} | RunnableAssign({'context' : context_getter})

## Chain 2 Specs: retrieval_chain -> generator_chain 
generator_chain = chat_prompt | llm  ## TODO
generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)  ## GIVEN

## END TODO
#####################################################################

rag_chain = retrieval_chain | generator_chain

# pprint(rag_chain.invoke("Tell me something interesting!"))
for token in rag_chain.stream("Tell me something interesting!"):
    print(token, end="")

import random

num_questions = 3
synth_questions = []
synth_answers = []

simple_prompt = ChatPromptTemplate.from_messages([('system', '{system}'), ('user', 'INPUT: {input}')])

for i in range(num_questions):
    doc1, doc2 = random.sample(docs, 2)
    sys_msg = (
        "Use the documents provided by the user to generate an interesting question-answer pair."
        " Try to use both documents if possible, and rely more on the document bodies than the summary."
        " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
        " DO NOT SAY: \"Here is an interesting question pair\" or similar. FOLLOW FORMAT!"
    )
    usr_msg = (
        f"Document1: {format_chunk(doc1)}\n\n"
        f"Document2: {format_chunk(doc2)}"
    )

    qa_pair = (simple_prompt | llm).invoke({'system': sys_msg, 'input': usr_msg})
    synth_questions += [qa_pair.split('\n\n')[0]]
    synth_answers += [qa_pair.split('\n\n')[1]]
    pprint2(f"QA Pair {i+1}")
    pprint2(synth_questions[-1])
    pprint(synth_answers[-1])
    print()

## TODO: Generate some synthetic answers to the questions above.
##   Try to use the same syntax as the cell above
rag_answers = []
for i, q in enumerate(synth_questions):
    ## TODO: Compute the RAG Answer
    rag_answer = rag_chain.invoke(q)
    rag_answers += [rag_answer]
    pprint2(f"QA Pair {i+1}", q, "", sep="\n")
    pprint(f"RAG Answer: {rag_answer}", "", sep='\n')


## TODO: Adapt this prompt for whichever LLM you're actually interested in using. 
## If it's llama, maybe system message would be good?
eval_prompt = ChatPromptTemplate.from_template("""INSTRUCTION: 
Evaluate the following Question-Answer pair for human preference and consistency.
Assume the first answer is a ground truth answer and has to be correct.
Assume the second answer may or may not be true.
[1] The second answer lies, does not answer the question, or is inferior to the first answer.
[2] The second answer is better than the first and does not introduce any inconsistencies.

Output Format:
[Score] Justification

{qa_trio}

EVALUATION: 
""")

pref_score = []

trio_gen = zip(synth_questions, synth_answers, rag_answers)
for i, (q, a_synth, a_rag) in enumerate(trio_gen):
    pprint2(f"Set {i+1}\n\nQuestion: {q}\n\n")

    qa_trio = f"Question: {q}\n\nAnswer 1 (Ground Truth): {a_synth}\n\n Answer 2 (New Answer): {a_rag}"
    pref_score += [(eval_prompt | llm).invoke({'qa_trio': qa_trio})]
    pprint(f"Synth Answer: {a_synth}\n\n")
    pprint(f"RAG Answer: {a_rag}\n\n")
    pprint2(f"Synth Evaluation: {pref_score[-1]}\n\n")