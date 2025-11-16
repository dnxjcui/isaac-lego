"""
Example script demonstrating OpenBCI cyton board assembly instruction system using backend.py.

This script performs RAG-based retrieval and multimodal generation to identify
where OpenBCI cyton board pieces should be placed based on instruction manual pages.
"""

from pathlib import Path
from PIL import Image
from backend import RAG, Model


SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in OpenBCI cyton board assembly. Your task is to help users identify "
    "where OpenBCI cyton board pieces should be placed based on the instruction manual pages provided. "
    "The first image is the OpenBCI cyton board piece the user is asking about. The following images are relevant "
    "pages from the instruction manual. Analyze the instruction manual pages to find where this piece "
    "should be placed, and provide multiple bounding boxes using <point_box> tags to highlight. "
    "Specifically, highlight the pin number and the exact location of the board pin that we need to connect the earpiece to. "
    "Include a description in the mention attribute explaining where we should place the pin in the FIRST image. "
)


if __name__ == "__main__":
    # Initialize RAG system for building vector database
    print("Initializing RAG system...")
    rag = RAG(gpu_id=1)
    
    # Initialize Model for inference
    print("Initializing Model for inference...")
    model = Model()
    
    # Build vector database from PDF
    pdf_path = "data/context/openbci_cyton_board_instructions.pdf"
    print(f"\nBuilding vector database from PDF: {pdf_path}")
    vector_db = rag.build_database_from_pdf(pdf_path, verify=False)
    print(f"Vector database built with {len(vector_db)} pages")
    
    # Load query image
    query_image_path = "data/images/openbci_example.jpg"
    if not Path(query_image_path).exists():
        raise FileNotFoundError(f"Image not found: {query_image_path}")
    
    print(f"\nLoading query image: {query_image_path}")
    query_image = Image.open(query_image_path).convert("RGB")
    print(f"Query image size: {query_image.size}")
    
    # Encode query image
    print("Encoding query image...")
    query_embedding = rag.encode_image(query_image, verify=False)
    
    # Search for similar pages
    k = 8
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
    user_prompt = "Which pin should this earpiece connect to?"
    
    # Get vision token from config
    vision_token = model.config.vision_token
    
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
    
    # Process output: extract response, bounding boxes, and create annotated image
    print("\nProcessing model output...")
    output = model.output(
        full_output=full_output,
        image=query_image,
        output_path="data/output_images/openbci_cyton_board_earpiece_annotated.png"
    )

    # Display results
    print("\n=== ASSISTANT RESPONSE ===\n")
    print(output['response'])
    print("\n" + "=" * 50)
    
    print(f"\nFound {len(output['bounding_boxes'])} bounding box(es)")
