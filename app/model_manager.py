import os
import json
import time
import logging
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from sklearn.metrics.pairwise import cosine_similarity

from app.cache_generator import CacheGenerator
from app.embeddings import Embeddings
from app.retrieval import Retrieval
from app.rag_generator import RAGGenerator

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)


class ModelManager:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        # load model
        self.model_name = model_name
        try:
            self._load_model()
            self._load_cache()
            self._init_cag()
            self._init_rag()
        except Exception:
            logger.exception("Failed to initialize ModelManager")
            raise

    def _safe_load_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {path}")
        except Exception:
            logger.exception(f"Unexpected error loading JSON file: {path}")
        return None

    def _load_model(self):
        """ load tokenizer and model using 4-bit quantization to speed up inference and reduce memory
        """
        try:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant,
                device_map="auto",
            )
            self.device = next(self.model.parameters()).device
            logger.info(f"Loaded model '{self.model_name}' on {self.device}")
        except Exception:
            logger.exception("Error loading model or tokenizer")
            raise

    def _load_cache(self):
        """get cache files and setup CAG attributes
        """
        cache_path = os.path.join("app", "all_OGS_embedded_docs.json")
        data = self._safe_load_json(cache_path)
        if not data:
            raise RuntimeError("Cache file is missing or invalid")
        idxs = data.get("cache_idxs", [])
        self.cache_docs = [data["documents"][i] for i in idxs]
        self.cache_embed = np.array([data["embeddings"][i] for i in idxs])
        self.cache_urls = [data["urls"][i] for i in idxs]

    def _init_cag(self):
        """ instantiate CAG """
        try:
            self.cag = CacheGenerator(
                self.cache_docs, self.model, self.tokenizer, self.device
            )
        except Exception:
            logger.exception("Error initializing CacheGenerator")
            raise

    def _init_rag(self):
        try:
            # setup RAG attributes
            self.data_embed = Embeddings("app/all_OGS_embedded_docs.json", import_emb=True)
            self.rag_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
            )
            self.retriever = Retrieval(self.data_embed)
            self.rag = RAGGenerator(self.rag_pipeline)
        except Exception:
            logger.exception("Error setting up RAG components")
            raise

    def query_handler(self, query, threshold=0.57, max_tokens=150, gen_method='Greedy', temperature=0.7, top_k=50):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        start_time = time.time()

        try:
            # calculate embedding for query and reshape for cosine sim
            query_embedding = self.data_embed.get_embedding(query).reshape(1, -1)

            similarity = cosine_similarity(query_embedding, self.cache_embed)[0]
            max_similarity = float(np.max(similarity))

            if max_similarity > threshold:
                logger.info(f"CAG path triggered (sim={max_similarity:.3f})")
                output = self.cag.query_responder(
                    query, self.tokenizer, self.model,
                    max_tokens, gen_method, temperature, top_k
                )
                # return urls where the similarity exceeds the threshold
                urls = [self.cache_urls[idx] for idx in np.where(similarity > threshold)[0] ]
                method = "CAG"
            else:
                logger.info(f"RAG path triggered (sim={max_similarity:.3f})")
                results = self.retriever.search(query, k=5)
                output = self.rag.summarize_abstract(results["documents"], query)[1]["content"]
                urls = results.get("urls", [])
                method = "RAG"
            print(method)
            return {
                "method": method,
                "output": output,
                "urls": urls,
                "cache_check_time": time.time() - start_time,
            }

        except Exception as e:
            logger.exception("Error in query_handler")
            return {
                "method": "ERROR",
                "output": "An internal error occurred. Please try again later.",
                "urls": [],
                "cache_check_time": time.time() - start_time,
            }



