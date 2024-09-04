from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from PIL import Image
import numpy as np
import os

# Define the model ID and revision
model_id = "vikhyatk/moondream2"
revision = "2024-04-02"

class ComfyUI_ImageToText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "log_prompt": (["No", "Yes"], {"default": "Yes"}),
            },
        }

    RETURN_TYPES = ('STRING', 'IMAGE')
    RETURN_NAMES = ('text_positive', 'image_preview')
    FUNCTION = "image2text"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI_Mexx"

    def check_model_availability(self, model_id, revision):
        # Check if the model exists locally
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
        model_path = os.path.join(model_cache_dir, model_id)

        if not os.path.exists(model_path):
            print(f"Model '{model_id}' not found locally. Attempting to download.")
        else:
            print(f"Model '{model_id}' found locally.")

    def generate_thumbnail(self, pil_image, size=(128, 128)):
        # Generate a thumbnail (resize) of the image for preview
        preview_image = pil_image.copy()
        preview_image.thumbnail(size)
        return preview_image

    def image2text(self, images, log_prompt):
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)
        
        # Ensure there's at least one valid image
        if len(pil_images) == 0 or pil_images[0].size == 0:
            raise ValueError("No valid images found. Ensure the input image is correctly loaded.")
        
        image = pil_images[0]
        
        # Create a preview thumbnail for the image
        image_preview = self.generate_thumbnail(image)

        # Check if the model is available
        self.check_model_availability(model_id, revision)
        
        # Load the model and tokenizer from Hugging Face
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}'. Ensure the model is available and correctly configured: {str(e)}")
        
        # Encode the image using the model
        try:
            enc_image = model.encode_image(image)
            if enc_image is None or enc_image.size == 0:
                raise ValueError("Encoded image is empty. Check the input image and model.")
            print(f"Encoded image size: {enc_image.size}")  # Debugging output
        except Exception as e:
            raise RuntimeError(f"Error encoding image: {str(e)}")
        
        # Generate text description from the image
        try:
            print(f"Trying to generate text description...")
            en = model.answer_question(enc_image, "Describe this image.", tokenizer)
        except AttributeError as e:
            raise AttributeError(f"Model '{model_id}' does not have 'answer_question' method: {str(e)}")
        except IndexError as e:
            print(f"Error: {str(e)}. This may be due to invalid position embeddings or input size.")
            print(f"Debugging position_ids or related tensor dimensions before the error.")
            # Debug the position_ids or similar tensor dimensions if possible here
            raise

        if log_prompt == "Yes":
            print(f"ImageToText: {en}")
        
        # Return the generated text and the preview image
        return [en, image_preview]

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComfyUI_ImageToText": ComfyUI_ImageToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_ImageToText": "ComfyUI_ImageToText"
}
