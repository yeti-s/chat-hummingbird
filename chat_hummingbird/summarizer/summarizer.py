import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer():
    def __init__(self, model_name:str, max_length:int=64, device=torch.device('cuda')) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device
        self.model.eval().to(device)
    
    # [summary, query, answer] -> summary
    @torch.no_grad()
    def summarize(self, content:list[str]) -> str:
        input_ids = self.tokenizer("[BOS]" + "[SEP]".join(content) + "[EOS]", return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        if 'token_type_ids' in input_ids:
            del input_ids['token_type_ids']
        gen_ids = self.model.generate(**input_ids, max_length=self.max_length, use_cache=True)
        generated = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return generated