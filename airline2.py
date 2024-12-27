#######################################################
# 1) Imports & Basic Python Setup
#######################################################
from pydantic import BaseModel, Field
from typing import Dict, Union, Optional, Iterable
from operator import itemgetter
import gradio as gr

# LangChain / Custom Imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough
)
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.messages import BaseMessage, SystemMessage, ChatMessage, AIMessage

#######################################################
# 2) Data & Model Definitions
#######################################################

class KnowledgeBase(BaseModel):
    """
    Holds relevant user & conversation info, including flight details.
    """
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: Optional[int] = Field(None, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: str = Field("", description="Topics that have not been resolved yet")
    current_goals: str = Field("", description="Current goal for the agent to address")

def get_flight_info(d: dict) -> str:
    """
    Simulated DB/lookup function that returns flight info based on name + confirmation.
    If no match, returns a string indicating not found.
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all(key in d for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
        ["Alice", "Johnson", 98765, "Chicago", "Miami", "7:00 PM", "11:00 PM", "next week"],
        ["Bob", "Brown", 56789, "Dallas", "Seattle", "1:00 PM", "4:00 PM", "yesterday"],
    ]

    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k: v for k, v in zip(keys, l)}
    db = {get_key(get_val(row)): get_val(row) for row in values}

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

def get_key_fn(base: BaseModel) -> dict:
    """
    Convert a KnowledgeBase object into a dictionary suitable for get_flight_info.
    """
    return {
        'first_name': base.first_name,
        'last_name': base.last_name,
        'confirmation': base.confirmation,
    }

#######################################################
# 3) Prompt Templates
#######################################################

# External prompt: final user-facing conversation prompt
external_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chatbot for SkyFlow Airlines, and you are helping a customer with their issue."
        " Please chat with them! Stay concise and clear!"
        " Your running knowledge base is: {know_base}."
        " This is for you only; Do not mention it!"
        " \nUsing that, we retrieved the following: {context}\n"
        " If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation."
        " Do not ask them any other personal info."
        " If it's not important to know about their flight, do not ask."
        " The checking happens automatically; you cannot check manually."
    ),
    ("assistant", "{output}"),  # The bot's previous turn
    ("user", "{input}"),        # The user’s new message
])

# Parser prompt: used by the "internal" chain to update knowledge base
parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the airline SkyFlow, and are trying to track info about the conversation."
    " You have just received a message from the user. Please fill in the schema based on the chat."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

#######################################################
# 4) LLM Setup & Chain Construction
#######################################################

# Create two LLMs:
# - chat_llm: used for final user-facing responses
# - instruct_llm: used for "extracting" / filling the knowledge base
chat_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

# External chain: From the external prompt -> final ChatNVIDIA response
external_chain = external_prompt | chat_llm

def RExtract(pydantic_class, llm, prompt):
    """
    Returns a Runnable that updates a Pydantic model (knowledge base) by extracting data from the conversation.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    print("Format instructions ", parser.get_format_instructions())
    instruct_merge = RunnableAssign({'format_instructions': lambda x: parser.get_format_instructions()})
    
    def preparse(string):
        # Minor cleanup to ensure JSON-like format
        if '{' not in string:
            string = '{' + string
        if '}' not in string:
            string = string + '}'
        string = (
            string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        return string
    
    # Chain steps: 1) inject format instructions 2) build prompt 3) LLM 4) cleanup 5) parse to KnowledgeBase
    return instruct_merge | prompt | llm | preparse | parser

# The chain that updates the knowledge base from user input + last assistant response
knowbase_getter = RExtract(KnowledgeBase, instruct_llm, parser_prompt)

# The chain that uses updated knowledge base to retrieve flight info
database_getter = (
    itemgetter('know_base')    # from state, pick out 'know_base'
    | RunnableLambda(get_key_fn) 
    | get_flight_info
)

# Combine into one "internal_chain":
# 1) Update knowledge base
# 2) Retrieve context from DB
internal_chain = (
    RunnableAssign({'know_base': knowbase_getter})
    | RunnableAssign({'context': database_getter})
)

#######################################################
# 5) The Chat Loop & Gradio Simulation
#######################################################

# Global state with an empty / default KnowledgeBase
state = {'know_base': KnowledgeBase()}

def chat_gen(message, history=[], return_buffer=True):
    """
    Main function that:
    1) Updates the knowledge base & context (internal_chain)
    2) Produces a new user-facing response (external_chain)
    3) Streams the result
    """
    global state
    state['input'] = message
    state['history'] = history
    state['output'] = "" if not history else history[-1][1]

    # 1) Run the internal chain (update knowledge base, get context)
    state = internal_chain.invoke(state)
    print("State after chain run:")
    print({k: v for k, v in state.items() if k != "history"})  # Debug printing

    # 2) Stream tokens from the external chain
    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token

def queue_fake_streaming_gradio(chat_stream, history=[], max_questions=8):
    """
    Simulates a Gradio-like streaming chat in the terminal.
    Each turn:
      - Ask user for input
      - Call chat_stream() to get the bot response
      - Print the response in real time
    """
    # Print existing history
    for human_msg, agent_msg in history:
        if human_msg:
            print("\n[ Human ]:", human_msg)
        if agent_msg:
            print("\n[ Agent ]:", agent_msg)

    # Chat loop
    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        
        # Collect streaming tokens
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        
        history += [history_entry]
        print("\n")

# Initialize a basic history with a greeting from the bot
chat_history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]

# Run the fake Gradio interface in console
if __name__ == "__main__":
    queue_fake_streaming_gradio(
        chat_stream=chat_gen,
        history=chat_history
    )
