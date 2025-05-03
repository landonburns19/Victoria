import sys
import os
import pickle
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from vector import retriever

# Set up model and prompt
model = OllamaLLM(model="llama3.2", max_tokens=12)

template = """
You are Victoria the ghost. You must provide dialogue that reflects your proclivities as a rich snob.

Here is the conversation so far:
{history}

Here are some relevant info: {reply}

Here is the dialogue to answer: {dialogue}
"""
prompt = ChatPromptTemplate.from_template(template)

# File to store memory
memory_file = "alonsomemory.pkl"

# Load existing memory or create new one
if os.path.exists(memory_file):
    with open(memory_file, "rb") as f:
        memory = pickle.load(f)
else:
    memory = ConversationBufferMemory(memory_key="history", input_key="dialogue")

# Set up the chain
chain = LLMChain(llm=model, prompt=prompt, memory=memory)

# Get dialogue from command line
dialogue = sys.argv[1] if len(sys.argv) > 1 else "say dot"

# Retrieve context and generate reply
reply = retriever.invoke(dialogue)
result = chain.invoke({"reply": reply, "dialogue": dialogue})

# Print result for Unity
print(result["text"])

# Save updated memory back to file
with open(memory_file, "wb") as f:
    pickle.dump(memory, f)
