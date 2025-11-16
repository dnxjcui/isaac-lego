import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from pdf2image import convert_from_path
from perceptron.tensorstream.ops import tensor_stream_token_view
from perceptron.pointing.parser import extract_points
from perceptron.pointing.geometry import scale_points_to_pixels
from rag import load_isaac_model, encode_images, encode_single_image, VectorDB, EmbeddingHook


SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in LEGO assembly. Your task is to help users identify "
    "where LEGO pieces should be placed based on the instruction manual pages provided. "
    "The first image is the LEGO piece the user is asking about. The following images are relevant "
    "pages from the instruction manual. Analyze the instruction manual pages to find where this piece "
    "should be placed, and provide multiple bounding boxes using <point_box> tags to highlight "
    "1. the location of the piece in the FIRST image and "
    "2. the location where we should place the piece in the FIRST image. "
    "Include a description in the mention attribute explaining where we should place the piece in the FIRST image."
)   

def load_model(device=None, dtype=None):
    """Load model using AutoProcessor (for inference with chat templates)."""
    hf_path = "PerceptronAI/Isaac-0.1"
    
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
    processor.tokenizer = tokenizer
    
    model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return config, tokenizer, processor, model, device, dtype


def extract_response(full_output: str) -> str:
    """Extract assistant response from full model output."""
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


if __name__ == "__main__":
    # Load model for inference
    config, tokenizer, processor, model, device, dtype = load_model()
    
    # Load model components for RAG (using IsaacProcessor)
    rag_model, rag_tokenizer, rag_processor, rag_device = load_isaac_model()
    
    # Build vector database from PDF
    pdf_path = "data/context/lego_instructions.pdf"
    print(f"Converting PDF: {pdf_path}")
    pdf_images = convert_from_path(pdf_path)
    print(f"Converted {len(pdf_images)} pages")
    
    print("Building vector database...")
    vector_db = encode_images(rag_model, rag_processor, pdf_images, rag_device, verify=False)
    print(f"Vector database built with {len(vector_db)} pages")
    
    # Load query image
    query_image_path = "data/images/bamboo_roma.jpg"
    if not Path(query_image_path).exists():
        raise FileNotFoundError(f"Image not found: {query_image_path}")
    
    print(f"\nLoading query image: {query_image_path}")
    query_image = Image.open(query_image_path).convert("RGB")
    print(f"Query image size: {query_image.size}")
    
    # Encode query image
    print("Encoding query image...")
    query_embedding = encode_single_image(rag_model, rag_processor, query_image, rag_device, verify=False)
    
    # Search for similar pages
    k = 3
    print(f"\nSearching for top {k} similar pages...")
    results = vector_db.search(query_embedding, k=k)
    
    print("\n=== SEARCH RESULTS ===")
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Page {result['metadata']['page']}, Similarity: {result['similarity']:.4f}")
    print("=" * 50)
    
    # Retrieve top-k images
    top_indices = [r['index'] for r in results]
    retrieved_images = [pdf_images[i] for i in top_indices]
    
    # Prepare messages with system prompt that specifically asks for bounding boxes
    user_prompt = "Where should this piece go?"
    
    # Build message list: system prompt, user query with query image, then retrieved pages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_prompt} {config.vision_token}"}
    ]
    
    # Add retrieved pages as additional context images
    all_images = [query_image] + retrieved_images
    for _ in range(len(retrieved_images)):
        messages.append({"role": "user", "content": config.vision_token})
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process with processor
    inputs = processor(text=text, images=all_images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    input_token_view = tensor_stream_token_view(tensor_stream)
    input_seq_len = input_token_view.shape[1] if len(input_token_view.shape) > 1 else input_token_view.shape[0]
    
    # Generate response
    print("\nGenerating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    # Decode full output
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    # Extract response text
    response = extract_response(full_output)
    
    print("\n=== ASSISTANT RESPONSE ===\n")
    print(response)
    print("\n" + "=" * 50)
    
    # Extract bounding boxes from the full output
    boxes = extract_points(full_output, expected="box")
    print(f"\nFound {len(boxes)} bounding box(es)")
    
    if len(boxes) == 0:
        print("No bounding boxes detected in the output.")
    else:
        # Convert normalized coordinates to pixel coordinates
        pixel_boxes = scale_points_to_pixels(boxes, width=query_image.width, height=query_image.height)
        
        # Create annotated image
        annotated_img = query_image.copy()
        draw = ImageDraw.Draw(annotated_img)
        # Portion of image width you want text width to be
        img_fraction = 0.15
        
        # Draw bounding boxes
        for i, box in enumerate(pixel_boxes):
            fontsize = 1  # starting font size
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", fontsize)
            top_left = (int(box.top_left.x), int(box.top_left.y))
            bottom_right = (int(box.bottom_right.x), int(box.bottom_right.y))
            
            # Draw rectangle
            draw.rectangle([top_left, bottom_right], outline="lime", width=20)
            
            # Add label with dynamic font sizing
            label = box.mention or f"location {i+1}"
            while font.getbbox(label)[2] < img_fraction * query_image.size[0]:
                # iterate until the text size is just larger than the criteria
                fontsize += 1
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", fontsize)
            draw.text((top_left[0], max(top_left[1] - font.getbbox(label)[3], 0)), label, fill="lime", font=font)
            
            print(f"\nBox {i+1}: {label}")
            print(f"  Top-left: {top_left}")
            print(f"  Bottom-right: {bottom_right}")
        
        # Save annotated image
        output_path = "data/images/bamboo_roma_annotated.png"
        annotated_img.save(output_path)
        print(f"\nSaved annotated image to: {output_path}")
