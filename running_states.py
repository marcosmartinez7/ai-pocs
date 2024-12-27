
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()

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




from langchain_core.runnables import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Union
from operator import itemgetter

## Zero-shot classification prompt and chain w/ explicit few-shot prompting
sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

zsc_prompt = ChatPromptTemplate.from_template(
    f"{sys_msg}\n\n"
    "[[The sea is awesome]][/INST]boat</s><s>[INST]"
    "[[{input}]]"
)

## Define your simple instruct_model
instruct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")
instruct_llm = instruct_chat | StrOutputParser()
one_word_llm = instruct_chat.bind(stop=[" ", "\n"]) | StrOutputParser()

zsc_chain = zsc_prompt | one_word_llm

## Function that just prints out the first word of the output. With early stopping bind
def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
    return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]

print("-" * 80)
print(zsc_call("Should I take the next exit, or keep going to the next one?"))

print("-" * 80)
print(zsc_call("I get seasick, so I think I'll pass on the trip"))

print("-" * 80)
print(zsc_call("I'm scared of heights, so flying probably isn't for me"))



gen_prompt = ChatPromptTemplate.from_template(
    "Make a new sentence about the the following topic: {topic}. Be creative!"
)

gen_chain = gen_prompt | instruct_llm

input_msg = "I get seasick, so I think I'll pass on the trip"
options = ["car", "boat", "airplane", "bike"]

chain = (
    ## -> {"input", "options"}
    {'topic' : zsc_chain}
    | PPrint()
    ## -> {**, "topic"}
    | gen_chain
    ## -> string
)

chain.invoke({"input" : input_msg, "options" : options})


from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.schema.runnable.passthrough import RunnableAssign
from functools import partial

big_chain = (
    PPrint()
    ## Manual mapping. Can be useful sometimes and inside branch chains
    | {'input' : lambda d: d.get('input'), 'topic' : zsc_chain}
    | PPrint()
    ## RunnableAssign passing. Better for running state chains by default
    | RunnableAssign({'generation' : gen_chain})
    | PPrint()
    ## Using the input and generation together
    | RunnableAssign({'combination' : (
        ChatPromptTemplate.from_template(
            "Consider the following passages:"
            "\nP1: {input}"
            "\nP2: {generation}"
            "\n\nCreate a text with words that are repeated from P1 and P2."
        )
        | instruct_llm
    )})
)

output = big_chain.invoke({
    "input" : "I get seasick, so I think I'll pass on the trip",
    "options" : ["car", "boat", "airplane", "bike", "unknown"]
})
pprint("Final Output: ", output)


