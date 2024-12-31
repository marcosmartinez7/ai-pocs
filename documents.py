from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()

def debug_print_prompt(state):
    """
    Prints the fully formatted prompt text that will be sent to the LLM.
    """
    # We assume the chain has placed the final prompt text in state["__prompt"] 
    # or something similar. If not, we'll see below how to intercept it.
    prompt_text = state.get("__prompt", "No prompt found in state")
    print("\n========== PROMPT DEBUG ==========")
    print(prompt_text)
    print("==================================\n")
    return state  # Return the state unchanged


## Useful utility method for printing intermediate states
from langchain_core.runnables import RunnableLambda
from functools import partial

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader

documents = ArxivLoader(query="2404.16130").load()  ## GraphRAG
#print("Number of Documents Retrieved:", len(documents))
# print(f"Sample of Document 1 Content (Total Length: {len(documents[0].page_content)}):")
#print(documents[0].page_content[:1000])
#pprint(documents[0].metadata)

from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

## Some nice custom preprocessing
documents[0].page_content = documents[0].page_content.replace(". .", "")
docs_split = text_splitter.split_documents(documents)

def include_doc(doc):
    ## Some chunks will be overburdened with useless numerical data, so we'll filter it out
    string = doc.page_content
    if len([l for l in string if l.isalpha()]) < (len(string)//2):
        return False
    return True

docs_split = [doc for doc in docs_split if include_doc(doc)]
print(len(docs_split))
for i in (0, 1, 2, 15, -1):
    pprint(f"[Document {i}]")
    print(docs_split[i].page_content)
    pprint("="*64)


from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from pydantic import BaseModel, Field
from typing import List
from IPython.display import clear_output
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})

    # This debug node will simply print the text
    debug_node = RunnableLambda(lambda text: (print("\n===== PROMPT DEBUG =====\n", text, "\n========================\n"), text)[1])

    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        return string
    
    # Chain steps:
    # 1) Add format instructions -> 2) build prompt -> 3) print the prompt -> 4) call LLM -> 5) preparse -> 6) parse
    return instruct_merge | prompt | debug_node | llm | preparse | parser




class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions that would be good to incorporate into summary, but that are yet unknown (max 3)")


summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one. Make sure a reader can still understand everything."
    " Keep it short, but as dense and useful as possible! The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keep all of the information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
)

latest_summary = ""

## TODO: Use the techniques from the previous notebook to complete the exercise
def RSummarizer(knowledge, llm, prompt, verbose=False):
    '''
    Exercise: Create a chain that summarizes
    '''
    ###########################################################################################
    ## START TODO:

    def summarize_docs(docs):        
        ## TODO: Initialize the parse_chain appropriately; should include an RExtract instance.
        ## HINT: You can get a class using the <object>.__class__ attribute...
        info_base_update = RExtract(knowledge.__class__, llm, prompt)

        parse_chain = RunnableAssign({'info_base' : info_base_update})
        ## TODO: Initialize a valid starting state. Should be similar to notebook 4
        state = {'info_base' : knowledge}

        global latest_summary  ## If your loop crashes, you can check out the latest_summary
        
        for i, doc in enumerate(docs):
            ## TODO: Update the state as appropriate using your parse_chain component
            state['input'] = doc.page_content

            state = parse_chain.invoke(state)

            assert 'info_base' in state 
            if verbose:
                print(f"Considered {i+1} documents")
                pprint(state['info_base'])
                latest_summary = state['info_base']
                clear_output(wait=True)

        return state['info_base']
        
    ## END TODO
    ###########################################################################################
    
    return RunnableLambda(summarize_docs)

# instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1").bind(max_tokens=4096)
instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1").bind(max_tokens=4096)
instruct_llm = instruct_model | StrOutputParser()

## Take the first 10 document chunks and accumulate a DocumentSummaryBase
summarizer = RSummarizer(DocumentSummaryBase(), instruct_llm, summary_prompt, verbose=True)
summary = summarizer.invoke(docs_split[:15])

pprint(latest_summary)