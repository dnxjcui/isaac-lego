from fastapi import FastAPI
from huggingface.modular_isaac import IsaacProcessor
from perceptron.tensorstream.ops import tensor_stream_token_view
import torch
from transformers import AutoTokenizer,

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "online"}

@app.post("/train")
def train_model(params: dict):
    # call your training script here
    return {"started": True}

@app.get("/process")
def process_prompt(params: dict):
    processor = params.processor
    retrieved_images = params.retrieved_images
    device = params.device
    model = params.model
    tokenizer = params.tokenizer

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
    return {"started": True}
