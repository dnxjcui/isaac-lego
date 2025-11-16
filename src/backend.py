"""
Backend module for LEGO assembly instruction system.

Provides RAG and Model classes for:
- Building and querying vector databases from PDF instruction manuals
- Loading and querying the Isaac model for multimodal generation
- Extracting bounding boxes and generating annotated images
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from pdf2image import convert_from_path
from perceptron.tensorstream.ops import tensor_stream_token_view
from perceptron.pointing.parser import extract_points
from perceptron.pointing.geometry import scale_points_to_pixels, BoundingBox
from src.huggingface.modular_isaac import IsaacProcessor


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
    
    def search(self, query_embedding: torch.Tensor, k: int = 3) -> List[Dict[str, Any]]:
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
    
    def get_by_page(self, page_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve embedding and metadata for a specific page number."""
        for idx, meta in enumerate(self.metadata):
            if meta.get('page') == page_number:
                return {
                    'index': idx,
                    'embedding': self.embeddings[idx],
                    'metadata': meta
                }
        return None
    
    def get_by_index(self, index: int) -> Optional[Dict[str, Any]]:
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


class RAG:
    """RAG (Retrieval-Augmented Generation) class for building and querying vector databases."""
    
    def __init__(self, gpu_id: int = 0):
        """
        Initialize RAG system.
        
        Args:
            gpu_id: GPU device ID to use
        """
        self.gpu_id = gpu_id
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load Isaac model for RAG operations."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            'PerceptronAI/Isaac-0.1', trust_remote_code=True, use_fast=False
        )
        config = AutoConfig.from_pretrained('PerceptronAI/Isaac-0.1', trust_remote_code=True)
        self.processor = IsaacProcessor(tokenizer=self.tokenizer, config=config)
        
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            'PerceptronAI/Isaac-0.1', trust_remote_code=True, torch_dtype=dtype
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def build_database_from_pdf(
        self, 
        pdf_path: str, 
        verify: bool = False
    ) -> VectorDB:
        """
        Build vector database from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            verify: Whether to print verification information for first image
        
        Returns:
            VectorDB instance containing encoded images
        """
        print(f"Converting PDF: {pdf_path}")
        pdf_images = convert_from_path(pdf_path)
        print(f"Converted {len(pdf_images)} pages")
        
        return self.encode_images(pdf_images, verify=verify)
    
    def encode_images(
        self, 
        images: List[Image.Image], 
        verify: bool = False
    ) -> VectorDB:
        """
        Encode images using Isaac vision encoder and store in VectorDB.
        
        Args:
            images: List of PIL Images to encode
            verify: Whether to print verification information for first image
        
        Returns:
            VectorDB instance containing encoded images
        """
        vector_db = VectorDB()
        hook = EmbeddingHook()
        hook.register(self.model)
        
        print(f"Encoding {len(images)} images through VisionEncoder...")
        for idx, img in enumerate(images):
            text = self.processor.vision_token
            inputs = self.processor(text=text, images=[img], return_tensors="pt")
            tensor_stream = inputs["tensor_stream"].to(self.device)
            
            with torch.no_grad():
                _ = self.model.model.embed_stream(tensor_stream)
            
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
    
    def encode_image(
        self, 
        image: Image.Image, 
        verify: bool = False
    ) -> torch.Tensor:
        """
        Encode a single image using Isaac vision encoder.
        
        Args:
            image: PIL Image to encode
            verify: Whether to print verification information
        
        Returns:
            Encoded embedding tensor
        """
        hook = EmbeddingHook()
        hook.register(self.model)
        
        text = self.processor.vision_token
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        tensor_stream = inputs["tensor_stream"].to(self.device)
        
        with torch.no_grad():
            _ = self.model.model.embed_stream(tensor_stream)
        
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
    
    def encode_text_query(
        self, 
        query: str, 
        verify: bool = False
    ) -> torch.Tensor:
        """
        Encode text query using Isaac text encoder.
        
        Args:
            query: Text query string
            verify: Whether to print verification information
        
        Returns:
            Encoded embedding tensor
        """
        tokens = self.tokenizer(query, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeds = self.model.model.embed_tokens(tokens.input_ids)
        
        query_embedding = embeds.mean(dim=1).squeeze(0)
        
        if verify:
            print(f"\n=== QUERY EMBEDDING VERIFICATION ===")
            print(f"Query: '{query}'")
            print(f"Embedding shape: {query_embedding.shape}")
            print(f"Embedding dtype: {query_embedding.dtype}")
            print(f"Sample values (first 10): {query_embedding[:10].tolist()}")
            print("=" * 50)
        
        return query_embedding


class Model:
    """Model class for loading and querying the Isaac model."""
    
    def __init__(self, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        """
        Initialize Model.
        
        Args:
            device: Device to use (defaults to cuda if available, else cpu)
            dtype: Data type to use (defaults to bfloat16 if cuda available, else float32)
        """
        self.device = device
        self.dtype = dtype
        self.config = None
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Isaac model using AutoProcessor."""
        hf_path = "PerceptronAI/Isaac-0.1"
        
        self.config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_path, trust_remote_code=True, use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype is None:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
    
    def query_model(
        self,
        messages: List[Dict[str, str]],
        images: List[Image.Image],
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.2,
        do_sample: bool = False
    ) -> str:
        """
        Query the model with messages and images.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            images: List of PIL Images to include in the query
            max_new_tokens: Maximum number of new tokens to generate
            repetition_penalty: Repetition penalty for generation
            do_sample: Whether to use sampling
        
        Returns:
            Full decoded output string
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process with processor
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        tensor_stream = inputs["tensor_stream"].to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        
        # Decode full output
        full_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return full_output
    
    def extract_response(self, full_output: str) -> str:
        """
        Extract assistant response from full model output.
        
        Args:
            full_output: Full decoded model output string
        
        Returns:
            Extracted response text
        """
        if '</think>' in full_output:
            parts = full_output.split('</think>')
            if len(parts) > 1:
                response = parts[-1].split('<|im_end|>')[0].strip()
            else:
                response = full_output.split('<|im_end|>')[0] if '<|im_end|>' in full_output else full_output
        elif '<|im_start|>assistant' in full_output:
            parts = full_output.split('<|im_start|>assistant')
            if len(parts) > 1:
                response = parts[-1].split('<|im_end|>')[0].strip()
                if '<think>' in response:
                    response = response.split('</think>')[-1].strip()
            else:
                response = full_output.split('<|im_end|>')[0] if '<|im_end|>' in full_output else full_output
        else:
            response = full_output.split('<|im_end|>')[0] if '<|im_end|>' in full_output else full_output
        
        response = response.lstrip('<|endoftext|>').strip()
        response = response.lstrip('\n').strip()
        return response
    
    def get_bounding_boxes(
        self, 
        full_output: str, 
        image_width: int, 
        image_height: int
    ) -> List[BoundingBox]:
        """
        Extract bounding boxes from model output and convert to pixel coordinates.
        
        Args:
            full_output: Full decoded model output string
            image_width: Width of the target image in pixels
            image_height: Height of the target image in pixels
        
        Returns:
            List of BoundingBox objects in pixel coordinates
        """
        boxes = extract_points(full_output, expected="box")
        if len(boxes) == 0:
            return []
        
        pixel_boxes = scale_points_to_pixels(
            boxes, width=image_width, height=image_height
        )
        return pixel_boxes
    
    def annotate_image(
        self,
        image: Image.Image,
        bounding_boxes: List[BoundingBox],
        output_path: Optional[str] = None,
        outline_color: str = "lime",
        outline_width: int = 20,
        text_color: str = "lime",
        img_fraction: float = 0.15,
        font_path: str = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
    ) -> Image.Image:
        """
        Annotate image with bounding boxes and labels.
        
        Args:
            image: PIL Image to annotate
            bounding_boxes: List of BoundingBox objects in pixel coordinates
            output_path: Optional path to save annotated image
            outline_color: Color for bounding box outlines
            outline_width: Width of bounding box outlines
            text_color: Color for text labels
            img_fraction: Fraction of image width for text sizing
            font_path: Path to font file
        
        Returns:
            Annotated PIL Image
        """
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        for i, box in enumerate(bounding_boxes):
            # Calculate font size dynamically
            fontsize = 1
            label = box.mention or f"location {i+1}"
            
            try:
                font = ImageFont.truetype(font_path, fontsize)
                while font.getbbox(label)[2] < img_fraction * image.size[0]:
                    fontsize += 1
                    font = ImageFont.truetype(font_path, fontsize)
            except Exception:
                # Fallback to default font if custom font not available
                font = ImageFont.load_default()
            
            top_left = (int(box.top_left.x), int(box.top_left.y))
            bottom_right = (int(box.bottom_right.x), int(box.bottom_right.y))
            
            # Draw rectangle
            draw.rectangle([top_left, bottom_right], outline=outline_color, width=outline_width)
            
            # Add label above the box
            text_y = max(top_left[1] - font.getbbox(label)[3], 0)
            draw.text((top_left[0], text_y), label, fill=text_color, font=font)
        
        if output_path:
            annotated_img.save(output_path)
        
        return annotated_img
    
    def output(
        self,
        full_output: str,
        image: Image.Image,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process model output: extract response, bounding boxes, and create annotated image.
        
        Args:
            full_output: Full decoded model output string
            image: PIL Image to annotate
            output_path: Optional path to save annotated image
        
        Returns:
            Dictionary containing:
                - response: Extracted text response
                - bounding_boxes: List of bounding box information
                - annotated_image: Annotated PIL Image
        """
        response = self.extract_response(full_output)
        bounding_boxes = self.get_bounding_boxes(
            full_output, image.width, image.height
        )
        
        annotated_image = None
        if len(bounding_boxes) > 0:
            annotated_image = self.annotate_image(
                image, bounding_boxes, output_path=output_path
            )
        
        box_info = []
        for i, box in enumerate(bounding_boxes):
            box_info.append({
                'index': i,
                'mention': box.mention or f"location {i+1}",
                'top_left': (int(box.top_left.x), int(box.top_left.y)),
                'bottom_right': (int(box.bottom_right.x), int(box.bottom_right.y))
            })
        
        return {
            'response': response,
            'bounding_boxes': box_info,
            'annotated_image': annotated_image
        }

