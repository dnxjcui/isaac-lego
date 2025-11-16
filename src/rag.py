import torch
import PIL.Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface.modular_isaac import IsaacProcessor
from perceptron.tensorstream.ops import tensor_stream_token_view
from pdf2image import convert_from_path

def load_isaac_model(gpu_id=0):
    """Load Isaac model."""
    tokenizer = AutoTokenizer.from_pretrained('PerceptronAI/Isaac-0.1', trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained('PerceptronAI/Isaac-0.1', trust_remote_code=True)
    processor = IsaacProcessor(tokenizer=tokenizer, config=config)
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained('PerceptronAI/Isaac-0.1', trust_remote_code=True, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer, processor, device


class EmbeddingHook:
    """Hook to extract vision embeddings from vision_embedding module."""
    def __init__(self):
        self.embeddings = None
        self.hook = None
    
    def _hook(self, module, input, output):
        self.embeddings = output.detach().clone()
    
    def register(self, model):
        self.hook = model.model.vision_embedding.register_forward_hook(self._hook)
    
    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
    
    def get_embeddings(self):
        return self.embeddings
    
    def clear(self):
        """Clear stored embeddings."""
        self.embeddings = None


class VectorDB:
    """In-memory vector database for storing and searching embeddings."""
    def __init__(self):
        self.embeddings = []
        self.metadata = []
    
    def add(self, embedding: torch.Tensor, metadata: dict = None):
        """Add an embedding with optional metadata."""
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def search(self, query_embedding: torch.Tensor, k: int = 3):
        """Search for top-k most similar embeddings using cosine similarity."""
        if len(self.embeddings) == 0:
            return []
        
        embeddings_tensor = torch.stack(self.embeddings)
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), embeddings_tensor, dim=1
        )
        top_k = torch.topk(similarities, k=min(k, len(similarities)))
        return [
            {
                'index': idx.item(),
                'similarity': sim.item(),
                'metadata': self.metadata[idx.item()]
            }
            for idx, sim in zip(top_k.indices, top_k.values)
        ]
    
    def get_by_page(self, page_number: int):
        """Retrieve embedding and metadata for a specific page number."""
        for idx, meta in enumerate(self.metadata):
            if meta.get('page') == page_number:
                return {
                    'index': idx,
                    'embedding': self.embeddings[idx],
                    'metadata': meta
                }
        return None
    
    def get_by_index(self, index: int):
        """Retrieve embedding and metadata by database index."""
        if 0 <= index < len(self.embeddings):
            return {
                'index': index,
                'embedding': self.embeddings[index],
                'metadata': self.metadata[index]
            }
        return None
    
    def __len__(self):
        return len(self.embeddings)


def encode_images(model, processor, images: list[PIL.Image.Image], device, verify: bool = True):
    """Encode images using Isaac vision encoder and store in VectorDB."""
    vector_db = VectorDB()
    hook = EmbeddingHook()
    hook.register(model)
    
    print(f"Encoding {len(images)} images through VisionEncoder...")
    for idx, img in enumerate(images):
        text = processor.vision_token
        inputs = processor(text=text, images=[img], return_tensors="pt")
        tensor_stream = inputs["tensor_stream"].to(device)
        
        with torch.no_grad():
            _ = model.model.embed_stream(tensor_stream)
        
        emb = hook.get_embeddings()
        if emb is not None:
            # Verify embeddings
            if verify and idx == 0:
                print(f"\n=== EMBEDDING VERIFICATION (first image) ===")
                print(f"Raw embedding shape: {emb.shape}")
                print(f"Raw embedding dtype: {emb.dtype}")
                print(f"Raw embedding device: {emb.device}")
                print(f"Sample values (first 10): {emb.flatten()[:10].tolist()}")
                print(f"Embedding stats - min: {emb.min().item():.6f}, max: {emb.max().item():.6f}, mean: {emb.mean().item():.6f}")
            
            # Average over sequence dimension to get single vector per image
            emb_mean = emb.mean(dim=0)
            
            if verify and idx == 0:
                print(f"\nAveraged embedding shape: {emb_mean.shape}")
                print(f"Sample values (first 10): {emb_mean[:10].tolist()}")
                print("=" * 50)
            
            vector_db.add(emb_mean, metadata={'page': idx, 'image': img})
        hook.clear()
    
    hook.remove()
    print(f"Stored {len(vector_db)} embeddings in vector database")
    return vector_db


def encode_single_image(model, processor, image: PIL.Image.Image, device, verify: bool = True):
    """Encode a single image using Isaac vision encoder."""
    hook = EmbeddingHook()
    hook.register(model)
    
    text = processor.vision_token
    inputs = processor(text=text, images=[image], return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    
    with torch.no_grad():
        _ = model.model.embed_stream(tensor_stream)
    
    emb = hook.get_embeddings()
    hook.remove()
    
    if emb is not None:
        # Average over sequence dimension to get single vector
        emb_mean = emb.mean(dim=0)
        
        if verify:
            print(f"\n=== IMAGE QUERY EMBEDDING VERIFICATION ===")
            print(f"Raw embedding shape: {emb.shape}")
            print(f"Averaged embedding shape: {emb_mean.shape}")
            print(f"Embedding dtype: {emb_mean.dtype}")
            print(f"Sample values (first 10): {emb_mean[:10].tolist()}")
            print("=" * 50)
        
        return emb_mean
    else:
        raise ValueError("Failed to extract embedding from image")


def encode_query(model, tokenizer, query: str, device, verify: bool = True):
    """Encode text query using Isaac text encoder."""
    tokens = tokenizer(query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeds = model.model.embed_tokens(tokens.input_ids)
    
    query_embedding = embeds.mean(dim=1).squeeze(0)
    
    if verify:
        print(f"\n=== QUERY EMBEDDING VERIFICATION ===")
        print(f"Query: '{query}'")
        print(f"Embedding shape: {query_embedding.shape}")
        print(f"Embedding dtype: {query_embedding.dtype}")
        print(f"Sample values (first 10): {query_embedding[:10].tolist()}")
        print("=" * 50)
    
    return query_embedding


def run_rag_pipeline(query: str, pdf_path: str = "data/lego_instructions.pdf", k: int = 3):
    """Run RAG pipeline: retrieve relevant pages and generate answer."""
    model, tokenizer, processor, device = load_isaac_model()
    
    # Convert PDF to images
    print(f"Converting PDF: {pdf_path}")
    pdf_images = convert_from_path(pdf_path)
    print(f"Converted {len(pdf_images)} pages")
    
    # Encode all images and store in vector database
    vector_db = encode_images(model, processor, pdf_images, device, verify=True)
    
    # Encode query
    query_embedding = encode_query(model, tokenizer, query, device, verify=True)
    
    # Search vector database
    print(f"\nSearching vector database for top {k} results...")
    results = vector_db.search(query_embedding, k=k)
    
    print(f"\n=== SEARCH RESULTS ===")
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Page {result['metadata']['page']}, Similarity: {result['similarity']:.4f}")
    print("=" * 50)
    
    # Retrieve top-k images
    top_indices = [r['index'] for r in results]
    retrieved_images = [pdf_images[i] for i in top_indices]
    print(f"Retrieved {len(retrieved_images)} pages")
    
    # Generate response
    print("Generating response...")
    vision_token = processor.vision_token
    text_parts = [f"Answer: {query}"]
    text_parts.extend([vision_token] * len(retrieved_images))
    text = " ".join(text_parts)
    
    inputs = processor(text=text, images=retrieved_images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    input_token_view = tensor_stream_token_view(tensor_stream)
    input_seq_len = input_token_view.shape[1] if len(input_token_view.shape) > 1 else input_token_view.shape[0]
    
    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    if generated_ids.shape[1] > input_seq_len:
        new_tokens = generated_ids[0, input_seq_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
        return response
    else:
        print("No response generated")
        return None


if __name__ == "__main__":
    query = "What are the assembly instructions?"
    
    # Load model and process PDF
    model, tokenizer, processor, device = load_isaac_model()
    pdf_path = "data/context/lego_instructions.pdf"
    
    print(f"Converting PDF: {pdf_path}")
    pdf_images = convert_from_path(pdf_path)
    print(f"Converted {len(pdf_images)} pages")
    
    # Encode all images and store in vector database
    vector_db = encode_images(model, processor, pdf_images, device, verify=True)
    
    # Look up page 17 specifically and use it as query image
    print(f"\n=== LOOKING UP PAGE 17 ===")
    page_17 = vector_db.get_by_page(17)
    if page_17:
        print(f"Found page 17 at index {page_17['index']}")
        print(f"Embedding shape: {page_17['embedding'].shape}")
        print(f"Metadata: {page_17['metadata']}")
        print("=" * 50)
        
        # Get the actual image for page 17
        page_17_image = page_17['metadata']['image']
        
        # Encode page 17 image as query embedding
        print(f"\n=== ENCODING PAGE 17 IMAGE AS QUERY ===")
        query_embedding = encode_single_image(model, processor, page_17_image, device, verify=True)
    else:
        print(f"Page 17 not found. Available pages: 0-{len(vector_db)-1}")
        print("=" * 50)
        raise ValueError("Page 17 not found in the database")
    
    # Search vector database using page 17's image embedding
    print(f"\nSearching vector database for top 3 results similar to page 17...")
    results = vector_db.search(query_embedding, k=3)
    
    print(f"\n=== SEARCH RESULTS (similar to page 17) ===")
    for i, result in enumerate(results, 1):
        page_num = result['metadata']['page']
        is_page_17 = " (QUERY PAGE)" if page_num == 17 else ""
        print(f"Rank {i}: Page {page_num}, Similarity: {result['similarity']:.4f}{is_page_17}")
    print("=" * 50)
    
    # Retrieve top-k images
    top_indices = [r['index'] for r in results]
    retrieved_images = [pdf_images[i] for i in top_indices]
    print(f"Retrieved {len(retrieved_images)} pages")
    
    # Generate response
    print("Generating response...")
    vision_token = processor.vision_token
    text_parts = ["Describe these images and how they relate to each other:"]
    text_parts.extend([vision_token] * len(retrieved_images))
    text = " ".join(text_parts)
    
    inputs = processor(text=text, images=retrieved_images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    input_token_view = tensor_stream_token_view(tensor_stream)
    input_seq_len = input_token_view.shape[1] if len(input_token_view.shape) > 1 else input_token_view.shape[0]
    
    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    if generated_ids.shape[1] > input_seq_len:
        new_tokens = generated_ids[0, input_seq_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
    else:
        print("No response generated")
