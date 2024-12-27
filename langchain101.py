## Necessary for Colab, not necessary for course environment
# %pip install -q langchain langchain-nvidia-ai-endpoints gradio

import os

## If you encounter a typing-extensions issue, restart your runtime and try again
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import langchain
langchain.verbose = False
#print(ChatNVIDIA.get_available_models())


from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

################################################################################
## Very simple "take input and return it"
identity = RunnableLambda(lambda x: x)  ## Or RunnablePassthrough works

################################################################################
## Given an arbitrary function, you can make a runnable with it
def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x

rprint0 = RunnableLambda(print_and_return)

################################################################################
## You can also pre-fill some of values using functools.partial
rprint1 = RunnableLambda(partial(print_and_return, preface="1: "))

################################################################################
## And you can use the same idea to make your own custom Runnable generator
def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

################################################################################
## Chaining two runnables
chain1 = identity | rprint1
chain1.invoke("Hello World!")
print()


################################################################################
## Chaining that one in as well
output = (
    chain1           ## Prints "Welcome Home!" & passes "Welcome Home!" onward
    | rprint1        ## Prints "1: Welcome Home!" & passes "Welcome Home!" onward
    | RPrint("2: ")  ## Prints "2: Welcome Home!" & passes "Welcome Home!" onward
).invoke("Welcome Home!")

## Final Output Is Preserved As "Welcome Home!"
print("\nOutput:", output)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Simple Chat Pipeline
chat_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in rhymes"),
    ("user", "{input}"),
])

rhyme_chain = prompt | chat_llm | StrOutputParser()

print(rhyme_chain.invoke({"input" : "Tell me about birds!"}))



import gradio as gr

def rhyme_chat_stream(message, history):
    ## This is a generator function, where each call will yield the next entry
    buffer = ""
    for token in rhyme_chain.stream({"input" : message}):
        buffer += token
        yield buffer

## Uncomment when you're ready to try this.
demo = gr.ChatInterface(rhyme_chat_stream).queue()
window_kwargs = {} # or {"server_name": "0.0.0.0", "root_path": "/7860/"}
# demo.launch(share=True, debug=True, **window_kwargs) 


## Feel free to try out some more models and see if there are better lightweight options
## https://build.nvidia.com
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")

sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

## One-shot classification prompt with heavy format assumptions.
zsc_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    ("user", "[[OMF red card!]]"),
    ("assistant", "soccer"),
    ("user", "[[{input}]]"),
])

zsc_chain = zsc_prompt | instruct_llm | StrOutputParser()

def zsc_call(input, options=["soccer", "tennis", "basketball", "golf"]):
    return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]

print("-" * 80)
print(zsc_call("That was a nice freekick"))

print("-" * 80)
print(zsc_call("Very long shot, scores 3 points"))

print("-" * 80)
print(zsc_call("Oh man, that one got stucked on the court"))

################################################################################
## Example of dictionary enforcement methods
def make_dictionary(v, key):
    if isinstance(v, dict):
        return v
    return {key : v}

def RInput(key='input'):
    '''Coercing method to mold a value (i.e. string) to in-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def ROutput(key='output'):
    '''Coercing method to mold a value (i.e. string) to out-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

################################################################################
## Common LCEL utility for pulling values from dictionaries
from operator import itemgetter

up_and_down = (
    RPrint("A: ")
    ## Custom ensure-dictionary process
    | RInput()
    | RPrint("B: ")
    ## Pull-values-from-dictionary utility
    | itemgetter("input")
    | RPrint("C: ")
    ## Anything-in Dictionary-out implicit map
    | {
        'word1' : (lambda x : x.split()[0]),
        'word2' : (lambda x : x.split()[1]),
        'words' : (lambda x: x),  ## <- == to RunnablePassthrough()
    }
    | RPrint("D: ")
    | itemgetter("word1")
    | RPrint("E: ")
    ## Anything-in anything-out lambda application
    | RunnableLambda(lambda x: x.upper())
    | RPrint("F: ")
    ## Custom ensure-dictionary process
    | ROutput()
)

up_and_down.invoke("Hello World")