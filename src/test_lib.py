from llm_toolkit.llm import LLM

llm = LLM("NousResearch/Llama-3.2-1B")
e = llm.embeddings("Hello, what's up?", 5)
print(e[:5])