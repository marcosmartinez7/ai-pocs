
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from copy import deepcopy

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")  ## Feel free to change the models

prompt1 = ChatPromptTemplate.from_messages([("user", (
    "INSTRUCTION: Only respond in rhymes"
    "\n\nPROMPT: {input}"
))])

prompt2 =  ChatPromptTemplate.from_messages([("user", (
    "INSTRUCTION: Only responding in rhyme, change the topic of the input poem to be about {topic}!"
    " Make it happy! Try to keep the same sentence structure, but make sure it's easy to recite!"
    " Try not to rhyme a word with itself."
    "\n\nOriginal Poem: {input}"
    "\n\nNew Topic: {topic}"
))])

## These are the main chains, constructed here as modules of functionality.
chain1 = prompt1 | instruct_llm | StrOutputParser()  ## only expects input
chain2 = prompt2 | instruct_llm | StrOutputParser()  ## expects both input and topic

################################################################################
## SUMMARY OF TASK: chain1 currently gets invoked for the first input.
##  Please invoke chain2 for subsequent invocations.

def rhyme_chat2_stream(message, history, return_buffer=True):
    '''This is a generator function, where each call will yield the next entry'''

    first_poem = None
    for entry in history:
        if entry[0] and entry[1]:
            ## If a generation occurred as a direct result of a user input,
            ##  keep that response (the first poem generated) and break out
            first_poem = "\n\n".join(entry[1].split("\n\n")[1:-1])
            break

    if first_poem is None:
        ## First Case: There is no initial poem generated. Better make one up!

        buffer = "Oh! I can make a wonderful poem about that! Let me think!\n\n"
        yield buffer

        ## Iterate over stream generator for first generation
        inst_out = ""
        chat_gen = chain1.stream({"input" : message})
        for token in chat_gen:
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token

        passage = "\n\nNow let me rewrite it with a different focus! What should the new focus be?"
        buffer += passage
        yield buffer if return_buffer else passage

    else:
        buffer = ""

        ## Subsequent Cases: There is a poem to start with. Generate a similar one with a new topic!
        chat_gen = chain2.stream({"input" : first_poem, "topic": message})
        inst_out = ""
        for token in chat_gen:
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token

        passage = "\n\nThis is fun! Give me another topic!"
        buffer += passage
        yield buffer if return_buffer else passage

################################################################################
## Below: This is a small-scale simulation of the gradio routine.

def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=3):

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
history = [[None, "Let me help you make a poem! What would you like for me to write?"]]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(
    chat_stream = rhyme_chat2_stream,
    history = history
)

## Simple way to initialize history for the ChatInterface
chatbot = gr.Chatbot(value = [[None, "Let me help you make a poem! What would you like for me to write?"]])

## IF USING COLAB: Share=False is faster
gr.ChatInterface(rhyme_chat2_stream, chatbot=chatbot).queue().launch(debug=True, share=True)