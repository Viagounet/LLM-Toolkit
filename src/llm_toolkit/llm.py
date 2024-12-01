from typing import Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
import time

from llm_toolkit.probe import ProbeModel





class LLM:
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    def generate_with_hook(self, prompt: str, layer: int, diff_in_m: Optional[torch.Tensor]=None):
        def activation_addition_hook(module, input, output):
            """
            Hook function to add the difference-in-means vector to activations.
            """
            if diff_in_m is None:
                return output
            if isinstance(output, tuple):
                output_vector = output[0]
            if output_vector.shape == (1,12,2048):
                return output
            return (output_vector + diff_in_m, output[1])
        target_layer = self.model.model.layers[layer]
        hook = target_layer.register_forward_hook(activation_addition_hook)
        max_length = 512
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            output_ids = self.model.generate(input_ids=input_ids, max_length=max_length)
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        finally:
            hook.remove()
        return generated_text
    
    def embeddings(self, prompt: str, layer: int) -> list[float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, 
                                            output_hidden_states=True, 
                                            return_dict_in_generate=True, 
                                            max_new_tokens=1, 
                                            min_new_tokens=1, 
                                            pad_token_id=self.tokenizer.pad_token_id)

        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        return last_hidden_state.numpy().tolist()

    def generation_tokens_embeddings(self, prompt: str, layer: int, max_new_token: int = 512) -> list[torch.Tensor]:
        """
        Returns a list of embeddings (at a certain layer) corresponding to the generated tokens
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, 
                                            output_hidden_states=True, 
                                            return_dict_in_generate=True, 
                                            max_new_tokens=max_new_token, 
                                            min_new_tokens=max_new_token, 
                                            pad_token_id=self.tokenizer.pad_token_id)
            
            pi_tokens_hidden_states_list: list[torch.Tensor] = []
            for token_hidden_state in outputs.hidden_states[1:]: # Iterating over post-instruction tokens
                pi_tokens_hidden_states_list.append(token_hidden_state[layer][0,0])
        return pi_tokens_hidden_states_list
    
    def step_generation(self, prompt: str, max_new_tokens: int = 512):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_ids = input_ids.clone()
        hidden_states_list = []
        with torch.no_grad():
            for _ in range(max_new_tokens):  # Specify the number of tokens to generate
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                ) # Returns logits, past_key_values & hidden states
                hidden_states = outputs.hidden_states[-1]
                last_token_representation = hidden_states[:, -1, :]
                print(last_token_representation)

                state_dict = self.model.state_dict()
                state_dict['model.layers.15.self_attn.o_proj.weight'] = state_dict['model.layers.0.self_attn.o_proj.weight'] + (torch.randn(state_dict['model.layers.15.self_attn.o_proj.weight'].size()) / 1000)
                self.model.load_state_dict(state_dict)
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                ) # Returns logits, past_key_values & hidden states

                hidden_states = outputs.hidden_states[-1]
                last_token_representation = hidden_states[:, -1, :]
                print(last_token_representation)

                input("\n\n")
                # logits = outputs.logits[:, -1, :]
                # next_token_id = torch.argmax(logits, dim=-1, keepdim=True) 
                # hidden_states_list.append(hidden_states)
                # generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                # attention_mask = attention_mask * 2
                # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long)], dim=-1)
                # token_string = self.tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
                # print(token_string)

    def train_probe(self, embeddings: list[np.ndarray], labels: list[str]):
        probe = ProbeModel(2048)
        # complete ...