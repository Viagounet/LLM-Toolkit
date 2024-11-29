from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
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