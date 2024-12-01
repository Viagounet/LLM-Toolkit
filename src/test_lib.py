import numpy as np
import torch
from llm_toolkit.llm import LLM
from sklearn.metrics.pairwise import cosine_similarity

llm = LLM("unsloth/Llama-3.2-1B-Instruct")
m1 = [llm.generation_tokens_embeddings("Harmful prompt 1", 10, max_new_token=512)] # [tensor.shape=(1x2048)]
m2 = [llm.generation_tokens_embeddings("Harmful prompt 2", 10, max_new_token=512)]
m3 = [llm.generation_tokens_embeddings("Harmful prompt 3", 10, max_new_token=512)]
m4 = [llm.generation_tokens_embeddings("Harmful prompt 4", 10, max_new_token=512)]
b1 = [llm.generation_tokens_embeddings("Harmless prompt 1", 10, max_new_token=512)]
b2 = [llm.generation_tokens_embeddings("Harmless prompt 2", 10, max_new_token=512)]
b3 = [llm.generation_tokens_embeddings("Harmless prompt 3", 10, max_new_token=512)]
b4 = [llm.generation_tokens_embeddings("Harmless prompt 4", 10, max_new_token=512)]

N_layers = 1
N_tokens = 64 - 1  # Removing one because indexing stops before max_new_token
embedding_dim = 2048

# Collect embeddings
all_m_embeddings = [m1, m2, m3, m4]
all_b_embeddings = [b1, b2, b3, b4]

def mu(post_instruction_embeddings_list):
    mean_embeddings = torch.zeros((N_layers, N_tokens, embedding_dim))
    # Iterate through layers and tokens to compute the mean embedding
    for l in range(N_layers):
        for t in range(N_tokens):
            embeddings = [m[l][t] for m in post_instruction_embeddings_list]
            mean_embeddings[l, t] = torch.mean(torch.stack(embeddings, dim=0), dim=0)
    return mean_embeddings

R = mu(all_m_embeddings) - mu(all_b_embeddings)
selected_r = R[0,0]
generated = llm.generate_with_hook("Your prompt", 10, -selected_r)