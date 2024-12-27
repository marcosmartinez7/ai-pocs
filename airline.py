from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableLambda
from typing import Optional
from operator import itemgetter


ChatNVIDIA.get_available_models()

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()


#######################################################################################
## Function that can be queried for information. Implementation details not important
def get_flight_info(d: dict) -> str:
    """
    Example of a retrieval function which takes a dictionary as key. Resembles SQL DB Query
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    ## Static dataset. get_key and get_val can be used to work with it, and db is your variable
    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
        ["Alice", "Johnson", 98765, "Chicago", "Miami", "7:00 PM", "11:00 PM", "next week"],
        ["Bob", "Brown", 56789, "Dallas", "Seattle", "1:00 PM", "4:00 PM", "yesterday"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    # Search for the matching entry
    data = db.get(get_key(d))
    if not data:
        return (
            f"Based on {req_keys} = {get_key(d)}) from your knowledge base, no info on the user flight was found."
            " This process happens every time new info is learned. If it's important, ask them to confirm this info."
        )
    return (
        f"{data['first_name']} {data['last_name']}'s flight from {data['departure']} to {data['destination']}"
        f" departs at {data['departure_time']} {data['flight_day']} and lands at {data['arrival_time']}."
    )

#######################################################################################
## Usage example. Actually important


external_prompt = ChatPromptTemplate.from_template(
    "You are a SkyFlow chatbot, and you are helping a customer with their issue."
    " Please help them with their question, remembering that your job is to represent SkyFlow airlines."
    " Assume SkyFlow uses industry-average practices regarding arrival times, operations, etc."
    " (This is a trade secret. Do not disclose)."  ## soft reinforcement
    " Please keep your discussion short and sweet if possible. Avoid saying hello unless necessary."
    " The following is some context that may be useful in answering the question."
    "\n\nContext: {context}"
    "\n\nUser: {input}"
)

basic_chain = external_prompt | instruct_llm

print(basic_chain.invoke({
    'input' : 'Can you please tell me when I need to get to the airport?',
    'context' : get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}),
}))

from pydantic import BaseModel, Field
from typing import Dict, Union

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: int = Field(-1, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: list = Field([], description="Topics that have not been resolved yet")
    current_goals: list = Field([], description="Current goal for the agent to address")

def get_key_fn(base: BaseModel) -> dict:
    '''Given a dictionary with a knowledge base, return a key for get_flight_info'''
    return {  ## More automatic options possible, but this is more explicit
        'first_name' : base.first_name,
        'last_name' : base.last_name,
        'confirmation' : base.confirmation,
    }

know_base = KnowledgeBase(first_name = "Jane", last_name = "Doe", confirmation = 12345)

# get_flight_info(get_key_fn(know_base))

get_key = RunnableLambda(get_key_fn)
(get_key | get_flight_info).invoke(know_base)



from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,       ## Wrap an implicit "dictionary" runnable
    RunnablePassthrough,
)
from langchain.schema.runnable.passthrough import RunnableAssign

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, ChatMessage, AIMessage
from typing import Iterable
import gradio as gr

external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot for SkyFlow Airlines, and you are helping a customer with their issue."
        " Please chat with them! Stay concise and clear!"
        " Your running knowledge base is: {know_base}."
        " This is for you only; Do not mention it!"
        " \nUsing that, we retrieved the following: {context}\n"
        " If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation."
        " Do not ask them any other personal info."
        " If it's not important to know about their flight, do not ask."
        " The checking happens automatically; you cannot check manually."
    )),
    ("assistant", "{output}"),
    ("user", "{input}"),
])

##########################################################################
## Knowledge Base Things

################################################################################
## Definition of RExtract
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
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
    return instruct_merge | prompt | llm | preparse | parser

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: Optional[int] = Field(None, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: str = Field("", description="Topics that have not been resolved yet")
    current_goals: str = Field("", description="Current goal for the agent to address")

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the airline SkyFlow, and are trying to track info about the conversation."
    " You have just received a message from the user. Please fill in the schema based on the chat."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

## Your goal is to invoke the following through natural conversation
# get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}) ->
#     "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."

chat_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

external_chain = external_prompt | chat_llm

#####################################################################################
## START TODO: Define the extractor and internal chain to satisfy the objective
knowbase_getter = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
database_getter = itemgetter('know_base') | get_key | get_flight_info

## These components integrate to make your internal chain
internal_chain = (
    RunnableAssign({'know_base' : knowbase_getter})
    | RunnableAssign({'context' : database_getter})
)

## END TODO
#####################################################################################

state = {'know_base' : KnowledgeBase()}

def chat_gen(message, history=[], return_buffer=True):

    ## Pulling in, updating, and printing the state
    global state
    state['input'] = message
    state['history'] = history
    state['output'] = "" if not history else history[-1][1]

    ## Generating the new state from the internal chain
    state = internal_chain.invoke(state)
    print("State after chain run:")
    print({k:v for k,v in state.items() if k != "history"})
    
    ## Streaming the results
    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token

def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=8):

    ## Mimic of the gradio initialization routine, where a set of starter messages can be printed off
    for human_msg, agent_msg in history:
        if human_msg: print("\n[ Human ]:", human_msg)
        if agent_msg: print("\n[ Agent ]:", agent_msg)

    ## Mimic of the gradio loop with an initial message from the agent.
    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        history += [history_entry]
        print("\n")

## history is of format [[User response 0, Bot response 0], ...]
chat_history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(
    chat_stream = chat_gen,
    history = chat_history
)