import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor

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


if __name__ == "__main__":
    config, tokenizer, processor, model, device, dtype = load_model()
    
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Use same pattern as reference code
    messages = [
        {"role": "user", "content": f"Describe this image in detail: {config.vision_token}"}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[dummy_image], return_tensors="pt")
    
    tensor_stream = inputs["tensor_stream"].to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode full output
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    # Extract assistant response
    # Pattern: content after </think> and before <|im_end|>
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
            # Remove redacted_reasoning block if present
            if '<think>' in response:
                response = response.split('</think>')[-1].strip()
        else:
            response = full_output.split('<|im_end|>')[0] if '<|im_end|>' in full_output else full_output
    else:
        # Fallback: try to find content before <|im_end|>
        response = full_output.split('<|im_end|>')[0] if '<|im_end|>' in full_output else full_output
    
    # Clean up: remove leading special tokens and whitespace
    response = response.lstrip('<|endoftext|>').strip()
    response = response.lstrip('\n').strip()
    
    print("\n=== EXTRACTED RESPONSE ===\n")
    print(response)

