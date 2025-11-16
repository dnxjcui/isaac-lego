import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from perceptron.pointing.parser import extract_points
from perceptron.pointing.geometry import scale_points_to_pixels


def load_model(device=None, dtype=None):
    """Load model using AutoProcessor (matches reference pattern)."""
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
    # Load model
    config, tokenizer, processor, model, device, dtype = load_model()
    
    # Load query image
    query_image_path = "data/images/bamboo_roma.jpg"
    if not Path(query_image_path).exists():
        raise FileNotFoundError(f"Image not found: {query_image_path}")
    
    query_image = Image.open(query_image_path).convert("RGB")
    print(f"Detecting green objects in: {query_image_path}")
    print(f"Image size: {query_image.size}")
    
    # Create prompt asking for green objects with bounding boxes
    prompt_text = (
        "Find every green object or green-colored item in this image. "
        "Return one bounding box per green object using <point_box> tags "
        "and include a description of what it is in the mention attribute."
    )
    messages = [
        {"role": "user", "content": f"{prompt_text} {config.vision_token}"}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[query_image], return_tensors="pt")
    
    tensor_stream = inputs["tensor_stream"].to(device)
    input_ids = inputs["input_ids"].to(device)
    
    # Generate response
    print("\nGenerating detection response...")
    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode full output
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    # Extract response text
    response = extract_response(full_output)
    print(f"\nDetection text:\n{response}\n")
    
    # Extract bounding boxes from the generated text
    boxes = extract_points(full_output, expected="box")
    print(f"Found {len(boxes)} green objects")
    
    if len(boxes) == 0:
        print("No green objects detected.")
    else:
        # Convert normalized coordinates to pixel coordinates
        pixel_boxes = scale_points_to_pixels(boxes, width=query_image.width, height=query_image.height)
        
        # Load image for drawing
        img = query_image.copy()
        draw = ImageDraw.Draw(img)
        # portion of image width you want text width to be
        img_fraction = 0.15

        for i, box in enumerate(pixel_boxes):
            # Draw bounding boxes
            fontsize = 1  # starting font size
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", fontsize)
            top_left = (int(box.top_left.x), int(box.top_left.y))
            bottom_right = (int(box.bottom_right.x), int(box.bottom_right.y))
            
            # Draw rectangle
            draw.rectangle([top_left, bottom_right], outline="lime", width=20)
            
            # Add label
            label = box.mention
            while font.getbbox(label)[2] < img_fraction*query_image.size[0]:
                # iterate until the text size is just larger than the criteria
                fontsize += 1
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", fontsize)
            draw.text((top_left[0], max(top_left[1] - font.getbbox(label)[3], 0)), label, fill="lime", font=font)
        
        # Save annotated image
        output_path = "data/images/bamboo_roma_annotated_2.png"
        img.save(output_path)
        print(f"\nSaved annotated image to: {output_path}")
