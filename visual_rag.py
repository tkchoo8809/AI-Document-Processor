import os
from time import time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1000"  # 120 seconds for each chunk
# os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")
# os.environ["HF_SKIP_CACHING_ALLOCATOR_WARMUP"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import json
import uuid, hashlib
from PIL import Image
from pathlib import Path
import torch, gc
from unsloth import FastVisionModel
from transformers import ColQwen2ForRetrieval, ColQwen2Processor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, ColPaliForRetrieval, ColPaliProcessor
from qwen_vl_utils import process_vision_info
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
from pydantic_classes import UniversalExtraction
import pypdfium2 as pdfium
import shutil
# from inference_comparison import evaluate_json
from difflib import SequenceMatcher
import math

class Qdrant:
    # Local Connection
    # def __init__(self, collection_name="visual_rag", host="localhost", port=6333):
    #     self.client = QdrantClient(host=host, port=port)
    #     self.collection_name = collection_name
    #     self._setup_collection()

    # Cloud Connection
    def __init__(self, collection_name=None, url=None, api_key=None):
        print("Connecting to Qdrant Cloud...")
        self.client = QdrantClient(
            url=url, 
            api_key=api_key,
            timeout=300,
            port=443,
            prefer_grpc=True,
            # port=6333, # local deployment
        )   
        self.collection_name = collection_name
        self._setup_collection()
        self.initialise_payload_indices()

    def _setup_collection(self):
        # ColQwen2 typically uses 128-dimensional vectors for its multi-vector patches
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Stage 1: Fast Dense Retrieval (Mean-Pooled)
                    "page_summary": models.VectorParams(
                        size=128, 
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
                    ),
                    # Stage 2: Precise Late Interaction (Multivector)
                    "colqwen_multivector": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0) # Indexing OFF for reranking
                    )
                }
            )

    def initialise_payload_indices(self):
        """
        Creates the necessary Qdrant indices for metadata filtering.
        Run this ONCE when setting up the database.
        """
        print("Creating payload indices...")
        
        # 1. Date of Issue (Keyword)
        # Required for: models.MatchValue(value="19-10-2023")
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="data.header.date_of_issue",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        # 2. Invoice Number (Keyword)
        # Required for: models.MatchValue(value="12345")
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="data.header.invoice_number",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        # 3. Seller Name (Text)
        # Required for: models.MatchText(text="Blue Spark")
        # "Text" index allows for partial/fuzzy matching (tokenization)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="data.header.seller_name",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD
            )
        )
        
        # 4. Buyer Name (Text)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="data.header.buyer_name",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD
            )
        )
    
    # def upsert(self, ids, multivectors, summary_vectors, metadata):
    #     # Convert tensors to numpy arrays once before uploading
    #     # This keeps the upload_collection call clean
    #     m_vecs = [m.detach().cpu().to(torch.float32).numpy() for m in multivectors]
    #     s_vecs = [s.detach().cpu().to(torch.float32).numpy() for s in summary_vectors]

    #     self.client.upload_collection(
    #         collection_name=self.collection_name,
    #         ids=ids,
    #         vectors={
    #             "page_summary": s_vecs,
    #             "colqwen_multivector": m_vecs
    #         },
    #         payload=metadata,
    #         max_retries=5,     # Automatically retry if connection glitches
    #     )

    def upsert(self, ids, multivectors, summary_vectors, metadata):
        # 1. Convert tensors to numpy once
        # We cast to list to ensure the PointStruct is JSON serializable for the client
        m_vecs = [m.squeeze().detach().cpu().to(torch.float32).numpy().tolist() for m in multivectors]
        s_vecs = [s.squeeze().detach().cpu().to(torch.float32).numpy().tolist() for s in summary_vectors]

        # 2. Prepare PointStruct list
        # zip allows us to iterate through all components simultaneously
        points = [
            PointStruct(
                id=idx,
                vector={
                    "page_summary": s_vec,
                    "colqwen_multivector": m_vec
                },
                payload=meta
            )
            for idx, m_vec, s_vec, meta in zip(ids, m_vecs, s_vecs, metadata)
        ]

        # 3. Perform the upsert
        # Chunking logic: Upload 10-20 points at a time for multi-vectors
        batch_size = 10 
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )

class VisualRAG:
    def __init__(self, qdrant_url, qdrant_api_key, qdrant_config=None, storage_dir="uploaded_images"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vl_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit" # "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        self.retriever_name = "vidore/colqwen2-v1.0-hf" # "vidore/colpali-v1.3-hf" "jinaai/jina-embeddings-v4"
        self.retriever_model, self.retriever_processor = self.initialise_hf_retriever_model()
        self.vl_model, self.vl_processor = self.initialise_unsloth_inference_model()
        # self.vl_model, self.vl_processor = self.initialise_hf_inference_model()
        
        # Local Qdrant setup
        # config = qdrant_config or {"collection_name": "docs_collection"}
        # self.db = Qdrant(**config)

        # Cloud Qdrant setup
        self.db = Qdrant(
            collection_name="docs_collection", 
            url=qdrant_url, 
            api_key=qdrant_api_key,
        )
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    """
    UNSLOTH MODELS | More models at https://huggingface.co/unsloth
    -----------------------------------------------------------------------------------------------------------------------
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
    -----------------------------------------------------------------------------------------------------------------------
    

    RETRIEVAL MODELS | More models at https://huggingface.co/models?pipeline_tag=visual-document-retrieval&sort=downloads
    -----------------------------------------------------------------------------------------------------------------------
    "vidore/colqwen2-v1.0-hf"                         # 128-dimensional vectors
    "jinaai/jina-embeddings-v4"                       # 128-dimensional vectors
    "vidore/colpali-v1.3-hf"                          # 128-dimensional vectors
    "nvidia/nemotron-colembed-vl-4b-v2"               # no linear projection leading to large 2560-dimensional space
    -----------------------------------------------------------------------------------------------------------------------
    """

    def initialise_unsloth_inference_model(self):
        print(f"Loading Unsloth Inference Model: {self.vl_name}...")                 
        model, tokeniser = FastVisionModel.from_pretrained(
            self.vl_name,
            load_in_4bit=True,
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
            device_map=self.device,
            attn_implementation="sdpa" # PyTorch's Scaled Dot Product Attention
        )
        model.config.vision_config.min_pixels = 256 * 28 * 28
        # model.config.vision_config.max_pixels = 512 * 28 * 28     # recommended
        # model.config.vision_config.max_pixels = 768 * 28 * 28     # very sharp
        model.config.vision_config.max_pixels = 1024 * 28 * 28      # for blurry
        
        FastVisionModel.for_inference(model)

        self.print_gpu_memory_usage(model=self.vl_name)

        return model, tokeniser
    
    def initialise_hf_inference_model(self):  
        print(f"Loading HuggingFace Inference Model: {self.vl_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 if your GPU supports it (RTX 30 series+)
            load_in_4bit_kv_cache=True, # cache KV
        )
        model = AutoModel.from_pretrained(
            self.retriever_name,
            # dtype=torch.float16,  # base model
            quantization_config=bnb_config,
            device_map=self.device,
            attn_implementation="sdpa", # PyTorch's Scaled Dot Product Attention
        )
        processor = AutoProcessor.from_pretrained(
            self.retriever_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28 
        )

        self.print_gpu_memory_usage(model=self.vl_name)

        return model, processor

    def initialise_hf_retriever_model(self):
        print(f"Loading HuggingFace Retrieval & Embedding Model: {self.retriever_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 if your GPU supports it (RTX 30 series+)
        )

        # Model
        if self.retriever_name in ["vidore/colqwen2-v1.0-hf"]:
            retriever = ColQwen2ForRetrieval.from_pretrained(
                self.retriever_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=self.device
            ).eval()
        elif self.retriever_name in ["vidore/colpali-v1.3-hf"]:
            retriever = ColPaliForRetrieval.from_pretrained(
                self.retriever_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=self.device
            ).eval()
        else:
            retriever = AutoModel.from_pretrained(
                self.retriever_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=self.device
            ).eval()

        # Processor
        if self.retriever_name in ["vidore/colqwen2-v1.0-hf"]:
            processor = ColQwen2Processor.from_pretrained(self.retriever_name)
        elif self.retriever_name in ["vidore/colpali-v1.3-hf"]:
            processor = ColPaliProcessor.from_pretrained(self.retriever_name)
        elif self.retriever_name in ["jinaai/jina-embeddings-v4"]:
            processor = None
        else:
            AutoProcessor.from_pretrained(self.retriever_name, use_fast=True)
        
        self.print_gpu_memory_usage(model=self.retriever_name)

        return retriever, processor

    def print_gpu_memory_usage(self, model=None):
        usage = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] {model} | ({usage:.2f} GB)")


    def clear_vram(self, label="Step", reset_stats=False):
        # Move any lingering local references to the void
        gc.collect()
        
        # Synchronize ensures the GPU is actually finished with current tasks
        # if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
            
        if reset_stats:
            torch.cuda.reset_peak_memory_stats()
        
        # Get free and total memory from the hardware level
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        
        # Convert bytes to GB
        free_gb = free_gpu / 1024**3
        total_gb = total_gpu / 1024**3
        used_gb = torch.cuda.memory_allocated() / 1024**3

        print(f"[VRAM] Used ({used_gb:.2f} GB) | Free: {free_gb:.2f} GB | Total: {total_gb:.2f} GB")
    
    def embed(self, inputs, mode="image"):
        """
        inputs: List of file paths (for images) OR list of strings (for queries).
        """

        def _standardize_embeddings(raw_embeddings):
            if isinstance(raw_embeddings, torch.Tensor):
                final_embeddings = raw_embeddings.detach().cpu()
                
            elif isinstance(raw_embeddings, list):
                cpu_tensors = [
                    e.detach().cpu() if isinstance(e, torch.Tensor) else torch.tensor(e).cpu() 
                    for e in raw_embeddings
                ]
                try:
                    final_embeddings = torch.stack(cpu_tensors)
                except RuntimeError:
                    # Fallback if tensors have varying sequence lengths
                    final_embeddings = cpu_tensors
            else:
                final_embeddings = torch.tensor(raw_embeddings).cpu()
                
            return final_embeddings

        start_time = time()
        print(f"[DEBUG] Generating {mode} embeddings for inputs...")

        if next(self.retriever_model.parameters()).device.type == 'cpu':
            print("[DEBUG] Moving Retrieval Model to GPU for embedding...")
            self.retriever_model.to(self.device)
        self.clear_vram("Pre-Embed", reset_stats=True)

        with torch.inference_mode():
            if mode == "image":
                # Open and standardise image size
                pil_images = [Image.open(p).convert("RGB") for p in inputs]
                    
                # if self.retriever_name == "vidore/colqwen2-v1.0-hf" or self.retriever_name == "vidore/colpali-v1.3-hf":    
                #     # Process and move to device
                #     batch_doc = self.retriever_processor.process_images(pil_images, return_tensors="pt").to(self.device)
                #     outputs = self.retriever_model(**batch_doc)
                    
                #     # Reassign embeddings to the CPU version
                #     final_embeddings = outputs.embeddings.detach().cpu()
                #     del batch_doc
                #     del outputs
                #     for img in pil_images:
                #         img.close() # Explicitly close file handles
                #     del pil_images
                
                if self.retriever_name in ["vidore/colqwen2-v1.0-hf", "vidore/colpali-v1.3-hf"]:    
                    # Process and move to device
                    batch_doc = self.retriever_processor.process_images(pil_images, return_tensors="pt").to(self.device)
                    
                    # Cast incoming floats to bfloat16
                    batch_doc = {
                        k: v.to(self.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.device)
                        for k, v in batch_doc.items()
                    }

                    # Force the internal SDPA engine to strictly use bfloat16 for all dynamically created masks
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.retriever_model(**batch_doc)
                        
                    embeddings = outputs.embeddings # Or the appropriate output attribute for ColPali

                    # 3. Standardize and cleanup
                    final_embeddings = _standardize_embeddings(embeddings)
                    
                    for img in pil_images:
                        img.close()
                    del pil_images
                    del batch_doc
                    del outputs
                    del embeddings

                elif self.retriever_name == "jinaai/jina-embeddings-v4":
                    # Jina handles file paths natively; no need for PIL or a separate processor
                    embeddings = self.retriever_model.encode_image(
                        images=inputs,
                        task="retrieval",
                        return_multivector=True # Required for late interaction matching
                    )
                    final_embeddings = _standardize_embeddings(embeddings)
                    del embeddings
                
                # elif self.retriever_name == "nvidia/nemotron-colembed-vl-4b-v2": 
                #     from torch.utils.data import DataLoader
                    
                #     # Patching to prevent multifork processing initialised by model
                #     original_init = DataLoader.__init__
                #     def safe_init(self, *args, **kwargs):
                #         kwargs['num_workers'] = 0
                #         original_init(self, *args, **kwargs)
                #     DataLoader.__init__ = safe_init

                #     embeddings = self.retriever_model.forward_images(images=pil_images, batch_size=1).to(self.device)
                #     final_embeddings = _standardize_embeddings(embeddings)

                #     DataLoader.__init__ = original_init
                    
                #     for img in pil_images:
                #         img.close()
                #     del pil_images
                #     del embeddings
                
            elif mode == "query":
                # if self.retriever_name in ["vidore/colqwen2-v1.0-hf", "vidore/colpali-v1.3-hf"]:
                #     # Process text queries directly
                #     batch_query = self.retriever_processor.process_queries(inputs).to(self.device)
                #     outputs = self.retriever_model(**batch_query)
                #     final_embeddings = outputs.embeddings.detach().cpu()
                #     del batch_query
                #     del outputs

                if self.retriever_name in ["vidore/colqwen2-v1.0-hf", "vidore/colpali-v1.3-hf"]:
                    # Process text queries directly
                    batch_query = self.retriever_processor(text=inputs, return_tensors="pt")
                    batch_query = {
                        k: v.to(self.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.device)
                        for k, v in batch_query.items()
                    }
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.retriever_model(**batch_query)
                        
                    embeddings = outputs.embeddings # Or the appropriate output attribute for ColPali

                    # Standardize and cleanup
                    final_embeddings = _standardize_embeddings(embeddings)

                    del batch_query
                    del outputs
                    del embeddings

                elif self.retriever_name == "jinaai/jina-embeddings-v4":
                    # Jina's built-in encode_text handles text queries
                    embeddings = self.retriever_model.encode_text(
                        texts=inputs,
                        task="retrieval",
                        prompt_name="query", 
                        return_multivector=True # Required for late interaction matching
                    )
                    final_embeddings = _standardize_embeddings(embeddings)
                    del embeddings

                # elif self.retriever_name == "nvidia/nemotron-colembed-vl-4b-v2":
                #     embeddings = self.retriever_model.forward_queries(inputs, batch_size=1)
                    
                #     # Standardize and cleanupdoe
                #     final_embeddings = _standardize_embeddings(embeddings)
                #     del embeddings

            else:
                raise ValueError("Mode must be 'image' or 'query'")
        
        # Offload the model to CPU immediately to free VRAM
        if next(self.retriever_model.parameters()).device.type == 'cuda':
            print("[DEBUG] Moving Retrieval Model to CPU to free up GPU for inference...")
            self.retriever_model.to("cpu")
        self.clear_vram("Post-Embed", reset_stats=True)
        end_time = time()
        print(f"[DEBUG] Embedding time: {end_time - start_time:.2f}s")

        return final_embeddings

    def build_filters(self, date: str = None, seller: str = None, invoice_id: str = None, buyer: str = None):
        conditions = []
        
        # Note: We use "data.field_name" because your JSON shows the fields are nested under "data"
        if date:
            conditions.append(
                models.FieldCondition(key="data.header.date_of_issue", match=models.MatchValue(value=date))
            )
        if invoice_id:
            conditions.append(
                models.FieldCondition(key="data.header.invoice_number", match=models.MatchValue(value=invoice_id))
            )
        if seller:
            # MatchText allows for partial matches (e.g. "Blue Spark" matches "Blue Spark Design")
            conditions.append(
                models.FieldCondition(key="data.header.seller_name", match=models.MatchText(text=seller))
            )
        if buyer:
            # MatchText allows for partial matches (e.g. "Blue Spark" matches "Blue Spark Design")
            conditions.append(
                models.FieldCondition(key="data.header.buyer_name", match=models.MatchText(text=buyer))
            )

        if not conditions:
            return None

        for index, condition in enumerate(conditions):
            print(f"[DEBUG] Filter {index}: {condition}")

        return models.Filter(must=conditions)
    
    def search(self, query_text, k=3, filter_conditions=None):
        """
        Performs Hybrid Search:
        1. Vector Search (Semantic)
        2. Filter Search (Metadata Constraints)
        """
        start_time = time()
        self.clear_vram("Pre-Search", reset_stats=True)
        print(f"[DEBUG] Searching for: '{query_text}")

        with torch.inference_mode():
            q_emb = self.embed([query_text], mode="query") # (1, tokens, dim)

        # Full token matrix for Stage 2
        full_multivector = q_emb[0].cpu().numpy()
        # Mean-pooled vector for Stage 1
        mean_pooled_vector = q_emb[0].mean(dim=0).cpu().numpy()

        results = self.db.client.query_points(
            collection_name=self.db.collection_name,
            # query=models.FusionQuery(fusion=models.Fusion.RRF),
            # THE PREFETCH (Stage 1)
            prefetch=[
                models.Prefetch(
                    query=mean_pooled_vector,
                    using="page_summary",
                    limit=10,
                    filter=filter_conditions
                )
            ],
            # THE RERANK (Stage 2)
            query=full_multivector,
            using="colqwen_multivector",
            limit=k,
            query_filter=filter_conditions
        ).points

        self.clear_vram("Post-Search", reset_stats=True)
        end_time = time()
        print(f"[DEBUG] Search time: {end_time - start_time:.2f}s")

        # return [res.payload["path"] for res in results] # return for reinference which is unnecessary
        return [res.payload for res in results]

    def extract_metadata(self, pil_image: Image.Image) -> dict:
        raw_response = self.inference(user_query="", images=[pil_image], json_format=True)
        print(f"[DEBUG] Raw reponse: {raw_response}")
        try:
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', raw_response, re.DOTALL)
            if not json_match:
                return {"header": {}, "items": [], "totals": {}}
            
            def print_dict_types(data, indent=1):
                """Recursively prints the keys and data types of a nested dictionary or list."""
                spacing = " " * indent
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"[DEBUG]{spacing}Field: '{key}' | Data Type: {type(value)}")
                        if isinstance(value, (dict, list)):
                            print_dict_types(value, indent + 2)
                            
                elif isinstance(data, list):
                    # To avoid spamming the console, just print the first item's structure
                    if len(data) > 0:
                        print(f"[DEBUG]{spacing}List Index [0] | Data Type: {type(data[0])}")
                        print_dict_types(data[0], indent + 2)
                        if len(data) > 1:
                            print(f"[DEBUG]{spacing}... ({len(data)-1} more items in list) ...")

            # Validate and immediately return the INNER data
            extracted = UniversalExtraction.model_validate_json(json_match.group(0))
            final_data = extracted.data.model_dump()
            # print_dict_types(final_data)

            def get_similarity(a, b):
                # 1. Handle Nones
                if a is None and b is None: return 1.0
                if a is None or b is None: return 0.0

                # 2. Number Equivalence (Compare mathematically, not as strings)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if math.isclose(float(a), float(b), rel_tol=1e-5):
                        return 1.0
                    else:
                        return 0.0
                        
                # 3. String comparison (Clean punctuation and compare)
                str_a = str(a).strip().lower().replace(",", "").replace(".", "")
                str_b = str(b).strip().lower().replace(",", "").replace(".", "")
                
                return SequenceMatcher(None, str_a, str_b).ratio()

            def evaluate_json(ext, gt, path="root"):
                """
                Recursively compares extracted JSON vs Ground Truth.
                Returns a dictionary of scores per field, plus a 'total_accuracy' key at the root level.
                """
                results = {}
                
                # 1. Dictionary Comparison
                if isinstance(gt, dict):
                    for key in gt:
                        new_path = f"{path}.{key}"
                        if not isinstance(ext, dict) or key not in ext:
                            results[new_path] = 0.0  # Missing field or ext is not a dict
                        else:
                            results.update(evaluate_json(ext[key], gt[key], new_path))
                
                # 2. List Comparison
                elif isinstance(gt, list):
                    for i, gt_item in enumerate(gt):
                        new_path = f"{path}[{i}]"
                        # Check if ext is a list and has this index
                        if isinstance(ext, list) and i < len(ext):
                            results.update(evaluate_json(ext[i], gt_item, new_path))
                        else:
                            results[new_path] = 0.0 # Missing row
                
                # 3. Leaf Node Comparison
                else:
                    score = 1.0 if ext == gt else get_similarity(ext, gt)
                    results[path] = score
                    
                # 4. Calculate Total Percentage
                if path == "root" and len(results) > 0:
                    # distinct_scores excludes any nested metadata if you add any later
                    score_values = [v for k, v in results.items() if isinstance(v, float)]
                    
                    if score_values:
                        avg_score = sum(score_values) / len(score_values)
                        results['total_accuracy'] = avg_score  # Store as 0.0 to 1.0
                        results['total_percentage'] = f"{avg_score * 100:.2f}%" # Readable string

                return results

            # # Evaluation against ground truth
            # with open("augmented_data/gt_batch1-0002.json", "r") as f:
            #     ground_truth = json.load(f)
            # evaluation = evaluate_json(extracted.model_dump(), ground_truth)
            # print(json.dumps(evaluation, indent=4))
            
            return final_data
        except Exception as e:
            return {"header": {}, "items": [], "totals": {}}

    # def upload_and_index(self, file_bytes: bytes):
    #     start_time = time()
    #     print("[DEBUG] Uploading and indexing new image...")
    #     """
    #     1. Receives raw bytes from a user upload.
    #     2. Converts to PIL for model processing.
    #     3. Saves to disk for permanent storage.
    #     """
    #     # Convert bytes to PIL Image
    #     pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
    #     # 1. AI Metadata Extraction (Using the PIL Image)
    #     ai_metadata = self.extract_metadata(pil_image)
        
    #     # 2. Permanent Storage
    #     img_id = str(uuid.uuid4())
    #     save_path = self.storage_dir / f"{img_id}.png"
    #     pil_image.save(save_path)
        
    #     # 3. Vector Embedding (Using the saved path or PIL Image)
    #     embeddings = self.embed([str(save_path)], mode="image")
    #     summary_vector = embeddings[0].mean(dim=0)

    #     # 4. Final Upsert
    #     point_id = int(uuid.uuid4().int >> 96)
    #     payload = {
    #         "path": str(save_path),
    #         **ai_metadata
    #     }
        
    #     self.db.upsert([point_id], embeddings, [summary_vector], [payload])

    #     end_time = time()
    #     print(f"[DEBUG] Upload and index: {end_time - start_time:.2f}s")
    
    def _is_already_indexed(self, point_id: str) -> bool:
            """Checks if a specific point ID already exists in the Qdrant database."""
            try:
                # Assuming self.db.client is your official QdrantClient instance
                # Replace `self.collection_name` with your actual collection variable
                results = self.db.client.retrieve(
                    collection_name=self.db.collection_name,
                    ids=[point_id]
                )
                # If retrieve returns a list with items, the point exists
                return len(results) > 0
            except Exception as e:
                print(f"[WARNING] Could not check database for duplicate: {e}")
                # Default to False so the upload continues if the DB check fails, 
                # or return True if you want to fail safely.
                return False

    def upload_and_index(self, file_path: str):
        """Standardizes the entry point for both single uploads and batch uploads."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"[ERROR] File not found: {file_path}")
            return f"Error: File {file_path} not found."

        # --- NEW DEDUPLICATION CHECK ---
        # 1. Generate the base deterministic ID
        base_file_id = self.generate_deterministic_id(str(path_obj))
        
        # 2. Determine what the point ID would be based on the file type
        if path_obj.suffix.lower() == ".pdf":
            # PDFs in your code are indexed starting with page_0
            check_hasher = hashlib.md5(f"{base_file_id}_page_0".encode())
            check_id = str(uuid.UUID(check_hasher.hexdigest()))
        else:
            # Images are indexed using the base file hash directly
            check_id = base_file_id

        # # 3. Search Qdrant to see if this exact point ID already exists
        # if self._is_already_indexed(check_id):
        #     print(f"[DEBUG] Skipping {path_obj.name} as it already exists in Qdrant database.")
        #     return f"Skip: Document {path_obj.name} is already indexed (UUID: {check_id})."

        if path_obj.suffix.lower() == ".pdf":
            success, point = self._handle_pdf(path_obj)
        else:
            success, point = self._handle_image(path_obj)

        if not success:
            return f"Failed to index {path_obj.name}: No valid metadata could be extracted."
        
        return f"Successfully indexed {path_obj.name}. Point ID (UUID): {point}"
    
    def generate_deterministic_id(self, file_path: str) -> str:
        """Generates a consistent UUID based on the file's contents."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read the file in chunks to handle large PDFs/Images safely
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                
        # Convert the MD5 hex string into a standard UUID format
        return str(uuid.UUID(hasher.hexdigest()))

    def _handle_pdf(self, path_obj: Path):
        start_time = time()
        
        pdf = pdfium.PdfDocument(str(path_obj))
        global_header = {}
        
        # Generate a base deterministic ID for the whole file
        base_file_id = self.generate_deterministic_id(str(path_obj))
        points = []

        for page_index in range(len(pdf)):
            # 1. Render and Save
            page = pdf[page_index]
            pil_image = page.render(scale=2.0).to_pil()
            save_path = self.storage_dir / f"{path_obj.stem}_p{page_index+1}.png"
            pil_image.save(save_path)

            # 2. EXTRACT:
            # On Page 1: Extract Header + Items + Totals
            # On other pages: You can choose to extract Items only or skip to save time.
            print(f"[DEBUG] Extracting data from Page {page_index + 1}...")
            page_data = self.extract_metadata(pil_image)

            data_block = page_data.get("data", page_data)
            invoice_block = data_block.get("invoice", data_block)

            current_header = invoice_block.get("header", {})
            current_items = invoice_block.get("items", [])
            current_totals = invoice_block.get("totals", {})

            if page_index == 0:
                global_header = current_header

            payload_data = {
                "header": global_header, # Inherit Page 1 header for all pages
                "items": current_items,
                "totals": current_totals
            }

            has_header = any(v for v in payload_data["header"].values() if v not in [None, "", 0.0])
            has_items = len(payload_data["items"]) > 0
            has_totals = any(v for v in payload_data["totals"].values() if v not in [None, "", 0.0])
            is_completely_empty = not (has_header or has_items or has_totals)

            if is_completely_empty:
                if page_index == 0:
                    print(f"[ERROR] Page 1 of {path_obj.name} failed extraction. Skipping PDF.")
                    if save_path.exists():
                        os.remove(save_path)
                    pdf.close()
                    return False, None
                else:
                    print(f"[ERROR] Page {page_index + 1} of {path_obj.name} is empty. Skipping this page.")
                    if save_path.exists():
                        os.remove(save_path)  
                    continue

            # 3. EMBED
            embeddings = self.embed([str(save_path)], mode="image")
            summary_vector = [embeddings[0].mean(dim=0)]

            # 4. PAYLOAD
            # We merge the Page 1 Header into every page's payload 
            # so searching for "Invoice #123" finds Page 2, 3, etc.
            payload = {
                "path": str(save_path),
                "filename": path_obj.name,
                "page": page_index + 1,
                "data": payload_data
            }

            # 5. UPSERT
            page_hasher = hashlib.md5(f"{base_file_id}_page_{page_index}".encode())
            page_uuid = str(uuid.UUID(page_hasher.hexdigest()))
            point_id = [page_uuid]
            self.db.upsert(point_id, embeddings, summary_vector, [payload])
            points.append(point_id)
            print(f"[DEBUG] Uploaded page {page_index + 1}: {time() - start_time:.2f}s")
            
        pdf.close()

        return True, points

    def _handle_image(self, path_obj: Path):
        start_time = time()
        
        save_path = self.storage_dir / path_obj.name
        if path_obj.absolute() != save_path.absolute():
            shutil.copy(path_obj, save_path)
        
        with Image.open(save_path) as img:
            # Extract everything: Header, Items, and Totals
            page_data = self.extract_metadata(img)

        data_block = page_data.get("data", page_data)
        invoice_block = data_block.get("invoice", data_block)

        # Check if the extraction is successful
        has_header = any(v for v in invoice_block.get("header", {}).values() if v not in [None, "", 0.0])
        has_items = len(invoice_block.get("items", [])) > 0
        has_totals = any(v for v in invoice_block.get("totals", {}).values() if v not in [None, "", 0.0])
        is_completely_empty = not (has_header or has_items or has_totals)

        if is_completely_empty:
            print(f"[ERROR] Extraction failed for {path_obj.name}, skipping as no metadata found or failed to conform to json format.")
            if save_path.exists():
                os.remove(save_path) # Clean up the file to save disk space
            return False, None
        
        embeddings = self.embed([str(save_path)], mode="image")
        summary_vector = [embeddings[0].mean(dim=0)]

        payload = [{
            "path": str(save_path),
            "filename": path_obj.name,
            "page": 1,
            "data": invoice_block # Contains header, items, and totals
        }]
        
        doc_uuid = self.generate_deterministic_id(str(path_obj))
        point_id = [doc_uuid]
        self.db.upsert(point_id, embeddings, summary_vector, payload)
       
        print(f"[DEBUG] Upload image: {time() - start_time:.2f}s")

        return True, point_id

    def inference(self, user_query, images=None, k=3, json_format=False):
        start_time = time()
        if next(self.retriever_model.parameters()).device.type == 'cuda':
            print("[DEBUG] Moving Retrieval Model to CPU to free up GPU for inference...")
            self.retriever_model.to("cpu")
        self.clear_vram("Pre-Inference", reset_stats=True)

        # 1. Identify mode and set prompt
        # is_text_only = len(images) == 0
        images = images or []

        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)

        if json_format:
            # Mode: Structured Data Extraction
            print(f"[DEBUG] Running JSON extraction inference...")
            schema_dict = UniversalExtraction.model_json_schema()
            json_schema_str = json.dumps(schema_dict, indent=2)
            
            prompt = (
                "You are an expert document parser. Your task is to extract data from the image "
                "into a structured JSON format. Follow this schema strictly:\n"
                f"```json\n{json_schema_str}\n```\n"
                "Return ONLY the valid JSON object. If a field is not found, use null or 0.0."
            )
        else:
            # Mode: Standard Natural Language Q&A
            print(f"[DEBUG] Running inference for query: '{user_query}'")
            prompt = (
                "You are a helpful assistant. Use the provided document images to answer the question.\n"
                "If the information is not in the images, say you don't know.\n\n"
                f"Question: {user_query}"
            )

        # 2. Build Content List
        content = []
        for img in images:
            # Ensure it's a PIL Image
            content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # 3. Process with Chat Template
        text = self.vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt"
        }
        
        # 4. Prepare Model Inputs
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
        inputs = self.vl_processor(**processor_kwargs).to(self.device)

        # 5. Generate with Strict Settings
        with torch.inference_mode():
            output_ids = self.vl_model.generate(
                **inputs, 
                max_new_tokens=2048 if json_format else 1024, # JSON needs more tokens
                use_cache=True, 
                temperature=0.01, # Keep it deterministic for JSON
            )
            
        generated_ids = [ids[len(inputs.input_ids[0]):] for ids in output_ids]
        response = self.vl_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 6. Cleanup VRAM and delete tensors
        del inputs
        del output_ids
        del generated_ids
        for img in loaded_images:
            if hasattr(img, "close"):
                img.close()
        del loaded_images

        self.clear_vram("Post-Inference", reset_stats=True)
        print(f"[DEBUG] Inference complete: {time() - start_time:.2f}s")

        return response

# def main():
#     print("Initializing Visual RAG...")
#     env_path = "credentials.env"
#     load_dotenv(dotenv_path=env_path)
#     QDRANT_URL = os.getenv("QDRANT_URL")
#     QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
#     rag = VisualRAG(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY)

#     # Test case 1: Upload and index images with user query
#     # rag.upload_and_index("augmented_data/batch2-0001_augmented.jpg")
#     result = rag.extract_metadata(Image.open("uploaded_images/batch1-0002_augmented.jpg").convert("RGB"))

    # # Test case 2: Embedding of image and text query
    # rag.embed(["augmented_data/batch2-0001_augmented.jpg"], mode="image")
    # rag.embed(["Extract invoice 257667"], mode="query")

    # # Test case 3: Search for specific invoice
    # targeted_filters = rag.build_filters(
    #                                     invoice_id="257667"
    #                                     )

    # rag.search(
    #             query_text="document", 
    #             k=3, 
    #             filter_conditions=targeted_filters
    #             )

# if __name__ == "__main__":
#     main()





