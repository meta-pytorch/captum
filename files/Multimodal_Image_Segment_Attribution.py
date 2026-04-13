import warnings

warnings.filterwarnings("ignore")

from io import BytesIO

import numpy as np
import requests
import torch
import transformers


# Captum imports for attribution
from captum.attr import FeatureAblation
from captum.attr._core.llm_attr import LLMAttribution
from captum.attr._utils.interpretable_input import ImageMaskInput
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")

# Model configuration
model_id = "google/gemma-4-31B-it"

# Load the model and processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"\nModel loaded successfully!")
print(f"Model device: {model.device}")

def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


# Load a sample image from HuggingFace
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image_from_url(image_url)

print(f"Image loaded successfully!")
print(f"Image size: {image.size} (width x height)")

# Display the image
image


# Define the conversation with image and text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Briefly describe the image in one sentence."},
        ],
    }
]

# Apply the chat template to format the prompt
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process the inputs (image + text) for the model
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
).to(model.device)

print("Input prepared successfully!")
print(f"Input token shape: {inputs['input_ids'].shape}")

# Generate the model's response
print("Generating response...")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # Greedy decoding for reproducibility
    )

# Decode only the generated tokens (exclude the input)
input_token_len = inputs["input_ids"].shape[1]
response = processor.decode(output[0][input_token_len:], skip_special_tokens=True)

print("\n" + "=" * 60)
print("Model Response:")
print("=" * 60)
print(response)


def create_grid_mask(image, rows, cols):
    """
    Create a grid mask that divides the image into rows x cols segments.
    
    Args:
        image: PIL Image
        rows: Number of rows in the grid
        cols: Number of columns in the grid
    
    Returns:
        Tensor of shape (height, width) with integer segment IDs
    """
    width, height = image.size
    mask = torch.zeros((height, width), dtype=torch.int32)
    
    h_step = height // rows
    w_step = width // cols
    
    for row in range(rows):
        for col in range(cols):
            # Calculate segment boundaries
            y_start = row * h_step
            y_end = height if row == rows - 1 else (row + 1) * h_step
            x_start = col * w_step
            x_end = width if col == cols - 1 else (col + 1) * w_step
            
            # Assign unique segment ID
            segment_id = row * cols + col
            mask[y_start:y_end, x_start:x_end] = segment_id
    
    return mask


# Create a 4x5 grid mask (20 segments)
grid_mask = create_grid_mask(image, rows=4, cols=5)
print(f"Grid mask shape: {grid_mask.shape}")
print(f"Number of segments: {len(torch.unique(grid_mask))}")


# Create ImageMaskInput with grid mask for VISUALIZATION ONLY
# We use a dummy processor (identity function) since we just want to preview the segmentation
# The actual ImageMaskInput for attribution will be created later with a proper processor_fn
grid_input_preview = ImageMaskInput(
    image=image,
    mask=grid_mask,
    processor_fn=lambda x: x,  # Dummy processor - just for visualization
)

print("Grid segmentation (4 rows × 5 columns = 20 segments):")
grid_input_preview.plot_mask_overlay(show=True)


from transformers import pipeline

# Load SAM-2 for automatic mask generation
sam_generator = pipeline(
    "mask-generation", model="facebook/sam2-hiera-large", device=0
)

print("SAM-2 model loaded successfully!")

# Generate semantic masks using SAM-2
# points_per_batch controls memory usage during mask generation
sam_outputs = sam_generator(image, points_per_batch=16)

# Extract the list of binary masks
sam_masks = sam_outputs["masks"]

print(f"SAM-2 generated {len(sam_masks)} semantic segments")
print(f"Each mask shape: {sam_masks[0].shape}")


# Similarly, create ImageMaskInput with SAM-2 masks for VISUALIZATION ONLY
# Notice how SAM-2 segments follow semantic boundaries (car body, windows, wheels, etc.)
sam_input_preview = ImageMaskInput(
    image=image,
    mask_list=sam_masks,
    processor_fn=lambda x: x,  # Dummy processor - just for visualization
)

print(f"SAM-2 segmentation ({len(sam_masks)} semantic segments):")
sam_input_preview.plot_mask_overlay(show=True)


# Initialize FeatureAblation with the model
fa = FeatureAblation(model)

# Wrap it with LLMAttribution for handling LLM-specific complexities
llm_attr = LLMAttribution(fa, processor.tokenizer)

print("Attribution tools initialized!")


def processor_fn(img):
    """
    Convert an image to model inputs.
    
    This function is called by ImageMaskInput for each perturbed image
    during the attribution process.
    
    Args:
        img: PIL Image (potentially with masked regions)
    
    Returns:
        Model inputs (tokenized text + processed image)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Briefly describe the image in one sentence."},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    return processor(
        text=prompt,
        images=img,
        return_tensors="pt",
    ).to(model.device)


print("Processor function defined!")


# Create ImageMaskInput with SAM-2 masks
sam_input = ImageMaskInput(
    image=image,
    mask_list=sam_masks,  # List of binary masks from SAM-2
    processor_fn=processor_fn,
)

print(f"Created ImageMaskInput with {sam_input.n_itp_features} interpretable features (segments)")

# Run attribution - this will ablate each segment and measure impact on output
print("\nRunning attribution (this may take a few minutes)...")
sam_attr_result = llm_attr.attribute(sam_input, forward_in_tokens=False)

print("Attribution complete!")


# Plot token-level attribution
# This shows which segments are important for each generated token
sam_attr_result.plot_token_attr(show=True)


# Plot overall image attribution heatmap
# This aggregates attribution across all generated tokens
sam_attr_result.plot_image_heatmap(show=True)


# First, let's see which tokens were generated
print("Generated tokens:")
for i, token in enumerate(sam_attr_result.output_tokens):
    print(f"  Position {i}: '{token}'")


# Example: Attribution heatmap for specific tokens
# Adjust the token positions based on the output above

# Heatmap for "Volkswagen Beetle" tokens
target_token_pos = (4, 6)
print(
    f'Showing heatmap for tokens at positions {target_token_pos} ["Volkswagen Beetle"]'
)
sam_attr_result.plot_image_heatmap(show=True, target_token_pos=target_token_pos)

# Heatmap for "two brown wooden doors" tokens (adjust positions as needed)
target_token_pos = (15, 19)
print(
    f'Showing heatmap for tokens at positions {target_token_pos} ["two brown wooden doors"]'
)
sam_attr_result.plot_image_heatmap(show=True, target_token_pos=target_token_pos)

# Create ImageMaskInput with grid mask for ATTRIBUTION (with proper processor_fn)
grid_input = ImageMaskInput(
    image=image,
    mask=grid_mask,
    processor_fn=processor_fn,  # Now using the real processor function
)

# Run attribution with grid-based segmentation
print(f"Running grid-based attribution with {grid_input.n_itp_features} segments...")
grid_attr_result = llm_attr.attribute(grid_input, forward_in_tokens=False)

print("Grid attribution complete!")

# Compare: visualize grid-based heatmap
print("\nGrid-based attribution heatmap:")
grid_attr_result.plot_image_heatmap(show=True)

