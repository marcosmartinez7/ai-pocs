
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

NVIDIAEmbeddings.get_available_models()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage
from operator import itemgetter

## Useful method for mistral, which is currently tuned to output numbered outputs
def EnumParser(*idxs):
    '''Method that pulls out values from a mistral model that outputs numbered entries'''
    idxs = idxs or [slice(0, None, 1)]
    entry_parser = lambda v: v if ('. ' not in v) else v[v.index('. ')+2:]
    out_lambda = lambda x: [entry_parser(v).strip() for v in x.split("\n")]
    return StrOutputParser() | RunnableLambda(lambda x: itemgetter(*idxs)(out_lambda(x)))

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1") | EnumParser()

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

gen_prompt = {'input' : lambda x:x} | ChatPromptTemplate.from_template(
    "Please generate 20 representative conversations that would be {input}."
    " Make sure all of the questions are very different in phrasing and content."
    " Do not respond to the questions; just list them. Make sure all of your outputs are numbered."
    " Example Response: 1. <question>\n2. <question>\n3. <question>\n..."
)

## Some that directly reference NVIDIA
responses_1 = (gen_prompt | instruct_llm).invoke(
    " reasonable for an NVIDIA document chatbot to be able to answer."
    " Vary the context to technology, research, deep learning, language modeling, gaming, etc."
)
print("Reasonable NVIDIA Responses:", *responses_1, "", sep="\n")

## And some that do not
responses_2 = (gen_prompt | instruct_llm).invoke(
    " be reasonable for a tech document chatbot to be able to answer. Make sure to vary"
    " the context to technology, research, gaming, language modeling, graphics, etc."
)
print("Reasonable non-NVIDIA Responses:", *responses_2, "", sep="\n")

## Feel free to try your own generations instead
responses_3 = (gen_prompt | instruct_llm).invoke(
    "unreasonable for an NVIDIA document chatbot to answer,"
    " as it is irrelevant and will not be useful to answer (though not inherently harmful)."
)
print("Irrelevant Responses:", *responses_3, "", sep="\n")

responses_4 = (gen_prompt | instruct_llm).invoke(
    "unreasonable for a chatbot (NVIDIA's, AMD's, Intels, or Generally) to answer,"
    " as an automated response will either be overly insensitive or offensive."
)
print("Harmful non-NVIDIA", *responses_4, "", sep="\n")

## Feel free to try your own generations instead

good_responses = responses_1 + responses_2
poor_responses = responses_3 + responses_4

import time
import numpy as np
import asyncio
from collections import abc
from typing import Callable
from functools import partial


class Timer():
    '''Useful timing utilities (%%time is great, but doesn't work for async)'''
    def __enter__(self):
      self.start = time.perf_counter()

    def __exit__(self, *args, **kwargs):
        elapsed = time.perf_counter() - self.start
        print("\033[1m" + f"Executed in {elapsed:0.2f} seconds." + "\033[0m")


async def embed_with_semaphore(
    text : str,
    embed_fn : Callable,
    semaphore : asyncio.Semaphore
) -> abc.Coroutine:
    async with semaphore:
        return await embed_fn(text)

## Making new embed method to limiting maximum concurrency
embed = partial(
    embed_with_semaphore,
    embed_fn = embedder.aembed_query,
    semaphore = asyncio.Semaphore(value=5)  ## <- feel free to play with value
)


good_tasks =  [embed(query) for query in good_responses]
poor_tasks = [embed(query) for query in poor_responses]
good_embeds = []
poor_embeds = []


async def main(embedder, good_responses):

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np

    with Timer():
        all_tasks = good_tasks + poor_tasks
        embeds = await asyncio.gather(*all_tasks)
        good_embeds = embeds[:len(good_tasks)]
        poor_embeds = embeds[len(good_tasks):]

        # Combine all groups into a single dataset
        embeddings = np.vstack([good_embeds, poor_embeds])

        # Labels for each point
        labels = np.array([0]*20 + [1]*20 + [4]*20 + [5]*20)

        # Perform PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Plotting PCA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels, cmap='viridis', label=labels)
        plt.title("PCA of Embeddings")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label='Group')

        # Plotting t-SNE
        plt.subplot(1, 2, 2)
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', label=labels)
        plt.title("t-SNE of Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.colorbar(label='Group')

        plt.show()

        

if __name__ == "__main__":
    results = asyncio.run(main(embedder, good_responses))


