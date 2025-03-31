import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import json
import numpy as np


# import Embeddings

class RefreshDocs:
    def __init__(self, action, model_name="microsoft/Phi-3.5-mini-instruct"):
        self.model_name = model_name
        if action == "CACHE":
            self.recalculate_cached_docs()
        else:
            self.recalculate_rag_embeddings()

    def load_model(self):
        # Load the tokenizer and model with 4-bit quantization to reduce memory usage.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map='auto'
        )
        self.device = self.model.model.embed_tokens.weight.device

    def recalculate_cached_docs(self, max_cache_tokens=int(128000 / 10)):  # only fill 1/x of 128k context
        data_embed = Embeddings('all_OGS_embedded_docs.json', import_emb=True)
        sorted_counts = np.argsort(data_embed.access_counts)[::-1]
        self.sorted_docs = [data_embed.data[i] for i in sorted_counts]
        self.sorted_embed = [data_embed.embeddings[i] for i in sorted_counts]
        self.load_model()

        total = 0
        self.cach_docs = []
        self.cach_emb = []
        for doc, emb in zip(self.sorted_docs, self.sorted_embed):
            doc_input_ids = self.tokenizer.encode(doc, return_tensors="pt").to(self.device)
            if (total + doc_input_ids.shape[1]) < max_cache_tokens:
                self.cach_docs.append(doc)
                self.cach_emb.append(emb)
                total += doc_input_ids.shape[1]
            else:
                break
        print('Tokens for Cache Docs', total)
        print('Number of Docs for Cache', len(self.cach_docs))
        with open('all_cache_docs.json', "w", encoding="utf-8") as f:
            json.dump({'embeddings': np.array(self.cach_emb).tolist(),
                       'documents': self.cach_docs
                       }, f, indent=4)

    def recalculate_rag_embeddings():
        # recalculate search embeddings, on some recurring frequency run:
        data_embed = Embeddings('all_document.txt')  # default arg import_emb=False so that this embeds documents

        # generate fake counts for testing, this needs some replacement for prod
        counts_log = np.clip(np.round(np.random.exponential(scale=5, size=len(data_embed.data))).astype(int), 0, 30)

        with open('all_OGS_embedded_docs.json', "w", encoding="utf-8") as f:
            json.dump({'embeddings': data_embed.embeddings.tolist(),
                       'documents': data_embed.data,
                       'access_count': counts_log.tolist()  # replace with some matching against the old list
                       }, f, indent=4)

# refresh = RefreshDocs(action="CACHE")