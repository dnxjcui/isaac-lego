# Source Code Documentation

This directory contains the core modules for the LEGO assembly instruction system using the Isaac multimodal model.

## Overview

The system uses Retrieval-Augmented Generation (RAG) to search through LEGO instruction manual pages and multimodal generation to identify where LEGO pieces should be placed.

## Core Modules

### `backend.py`

Main backend module providing high-level classes for RAG and model operations.

#### Classes

- **`RAG`**: Handles building and querying vector databases from PDF instruction manuals
  - `build_database_from_pdf()`: Convert PDF to images and build vector database
  - `encode_images()`: Encode multiple images and store in vector database
  - `encode_image()`: Encode a single image for querying
  - `encode_text_query()`: Encode text queries for semantic search

- **`Model`**: Handles loading and querying the Isaac model for multimodal generation
  - `query_model()`: Generate responses from messages and images
  - `extract_response()`: Extract assistant response from full model output
  - `get_bounding_boxes()`: Extract bounding boxes from model output
  - `annotate_image()`: Draw bounding boxes on images with labels
  - `output()`: Complete pipeline: extract response, bounding boxes, and create annotated image

- **`VectorDB`**: In-memory vector database for storing and searching embeddings
  - `add()`: Add embedding with metadata
  - `search()`: Search for top-k similar embeddings using cosine similarity
  - `get_by_page()`: Retrieve embedding by page number
  - `get_by_index()`: Retrieve embedding by database index

- **`EmbeddingHook`**: Hook to extract vision embeddings from the model's vision_embedding module

### `rag.py`

Lower-level RAG utilities and functions for encoding images and text queries. Provides functions for:
- Loading Isaac model components
- Encoding images and text queries
- Building vector databases
- Running RAG pipelines

**Note**: The `backend.py` module provides a higher-level interface. New code should prefer using `backend.py` classes.

### `example.py`

Example script demonstrating the complete workflow using `backend.py`:
1. Build vector database from PDF instruction manual
2. Load and encode a query image (LEGO piece)
3. Search for similar pages in the instruction manual
4. Generate multimodal response with bounding boxes
5. Create annotated image showing where pieces should be placed

**Usage**:
```bash
python src/example.py
```

### `get_instructions.py`

Reference implementation using lower-level functions from `rag.py`. This script demonstrates the same workflow as `example.py` but uses direct function calls instead of the `backend.py` classes.

### `detect_object.py`

Object detection utilities for extracting bounding boxes and creating annotated images. Provides helper functions for:
- Loading models
- Extracting responses
- Drawing bounding boxes on images

### `test_inference.py`

Simple test script for verifying model loading and basic inference functionality.

### `download_model.py`

Utility script for downloading and saving the Isaac model locally.

## Dependencies

- `torch`: PyTorch for model operations
- `transformers`: Hugging Face transformers library
- `PIL` (Pillow): Image processing
- `pdf2image`: PDF to image conversion
- `perceptron`: PerceptronAI libraries for tensor streams and pointing geometry

## Workflow

The typical workflow for using the system:

1. **Initialize RAG system**:
   ```python
   from backend import RAG
   rag = RAG(gpu_id=0)
   ```

2. **Build vector database**:
   ```python
   vector_db = rag.build_database_from_pdf("path/to/instructions.pdf")
   ```

3. **Encode query image**:
   ```python
   query_image = Image.open("query.jpg")
   query_embedding = rag.encode_image(query_image)
   ```

4. **Search for similar pages**:
   ```python
   results = vector_db.search(query_embedding, k=3)
   ```

5. **Initialize Model and generate response**:
   ```python
   from backend import Model
   model = Model()
   full_output = model.query_model(messages, images)
   output = model.output(full_output, query_image)
   ```

## File Structure

```
src/
├── backend.py          # Main backend classes (RAG, Model, VectorDB)
├── rag.py             # Lower-level RAG utilities
├── example.py         # Example using backend.py
├── get_instructions.py # Reference implementation using rag.py
├── detect_object.py   # Object detection utilities
├── test_inference.py  # Model inference tests
├── download_model.py  # Model download utility
└── README.md         # This file
```

## Notes

- The system uses the `PerceptronAI/Isaac-0.1` model from Hugging Face
- GPU acceleration is recommended but CPU fallback is supported
- Vector database stores images in memory - for large PDFs, consider disk-based storage
- Bounding boxes are extracted using the `perceptron.pointing` module

