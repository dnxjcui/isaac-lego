from pathlib import Path
from PIL import Image
from src.backend import RAG, Model
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from io import BytesIO
from fastapi.encoders import jsonable_encoder
import base64

app = FastAPI()
rag = None
model = None
vector_db = None
SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in LEGO assembly. Your task is to help users identify "
    "where LEGO pieces should be placed based on the instruction manual pages provided. "
    "The first image is the LEGO piece the user is asking about. The following images are relevant "
    "pages from the instruction manual. Analyze the instruction manual pages to find where this piece "
    "should be placed, and provide multiple bounding boxes using <point_box> tags to highlight "
    "1. the location of the piece in the FIRST image and "
    "2. the location from the FIRST image where we should place the piece. "
    "Include a description in the mention attribute explaining where we should place the piece in the FIRST image."
)

class TrainData(BaseModel):
    features: list[int]
    label: str


@app.get("/ping")
def ping():
    return {"status": "online"}

@app.get("/load_model")
def load_model():
    global rag, model, vector_db
    # Initialize RAG system for building vector database
    print("Initializing RAG system...")
    rag = RAG(gpu_id=0)
    
    # Initialize Model for inference
    print("Initializing Model for inference...")
    model = Model()
    
    # Build vector database from PDF
    pdf_path = "data/context/lego_instructions.pdf"
    print(f"\nBuilding vector database from PDF: {pdf_path}")
    vector_db = rag.build_database_from_pdf(pdf_path, verify=False)
    print(f"Vector database built with {len(vector_db)} pages")
    return {"started": True}

@app.post("/process")
async def process_prompt(
    text: str = Form(...),        
    picture: UploadFile = File(...)
    ):
    print("here1")

     # Read the file content
    contents = await picture.read()
    print("here2")
    query_image = Image.open(BytesIO(contents)).convert("RGB")
    print("here3")

    print(f"Query image size: {query_image.size}")
    
    # Encode query image
    print("Encoding query image...")
    query_embedding = rag.encode_image(query_image, verify=False)
    
    # Search for similar pages
    k = 3
    print(f"\nSearching for top {k} similar pages...")
    results = vector_db.search(query_embedding, k=k)
    
    print("\n=== SEARCH RESULTS ===")
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Page {result['metadata']['page']}, Similarity: {result['similarity']:.4f}")
    print("=" * 50)
    
    # Retrieve top-k images from vector database
    retrieved_images = []
    for result in results:
        page_data = vector_db.get_by_index(result['index'])
        if page_data and 'image' in page_data['metadata']:
            retrieved_images.append(page_data['metadata']['image'])
    
    # Prepare messages with system prompt
    user_prompt = "Where should this piece go?"
    
    # Get vision token from config
    vision_token = model.config.vision_token
    print("here4")
    
    # Build message list: system prompt, user query with query image, then retrieved pages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_prompt} {vision_token}"}
    ]
    
    # Add retrieved pages as additional context images
    all_images = [query_image] + retrieved_images
    for _ in range(len(retrieved_images)):
        messages.append({"role": "user", "content": vision_token})

    
    # Generate response using Model class
    print("\nGenerating response...")
    full_output = model.query_model(
        messages=messages,
        images=all_images,
        max_new_tokens=512,
        repetition_penalty=1.2,
        do_sample=False
    )
    print("here5")
    
    # Process output: extract response, bounding boxes, and create annotated image
    print("\nProcessing model output...")
    output = model.output(
        full_output=full_output,
        image=query_image,
        output_path="data/images/bamboo_roma_annotated.png"
    )
    
    # Display results
    print("\n=== ASSISTANT RESPONSE ===\n")
    print(output['response'])
    print("\n" + "=" * 50)
    
    print(f"\nFound {len(output['bounding_boxes'])} bounding box(es)")
    
    if len(output['bounding_boxes']) == 0:
        print("No bounding boxes detected in the output.")
    else:
        for i, box_info in enumerate(output['bounding_boxes'], 1):
            print(f"\nBox {i}: {box_info['mention']}")
            print(f"  Top-left: {box_info['top_left']}")
            print(f"  Bottom-right: {box_info['bottom_right']}")
        
        if output['annotated_image']:
            print(f"\nSaved annotated image to: data/images/bamboo_roma_annotated.png")


    # Serialize image to base64 string
    annotated_image_b64 = None
    if output.get("annotated_image"):
        buffered = BytesIO()
        output["annotated_image"].save(buffered, format="PNG")
        annotated_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Serialize bounding boxes and replace image
    safe_output = {
        "bounding_boxes": output["bounding_boxes"],
        "annotated_image": annotated_image_b64
    }

    return {"started": True, "output": jsonable_encoder(safe_output)}
