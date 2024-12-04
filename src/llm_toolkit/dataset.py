import torch
from typing import List, Dict, Any

class DataCollator:
    def __init__(self, embedding_function, target_layer: int, device='auto'):
        """
        Args:
            embedding_function: A function that converts text to embeddings (returns a torch tensor).
            device: The device to move tensors to ('cpu' or 'cuda').
        """
        self.embedding_function = embedding_function
        self.device = device
        self.target_layer = target_layer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples.
        Args:
            batch: A list of dictionaries, each containing 'text' and 'label'.

        Returns:
            A dictionary with 'input_ids' and 'labels' as tensors.
        """
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Generate embeddings for each text
        embeddings = torch.stack([torch.Tensor(self.embedding_function(text, layer=self.target_layer)) for text in texts])
        
        # Convert labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Move tensors to the specified device
        return {
            'input_ids': embeddings.to(self.device),
            'labels': labels_tensor.to(self.device)
        }

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}
    
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}
