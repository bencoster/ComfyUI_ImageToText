from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

class ComfyUI_ImageToText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "log_prompt": (["No", "Yes"], {"default":"Yes"}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text_positive',)
    FUNCTION = "image2text"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI_Mexx"

    def check_model_availability(self, model_id, revision):
        """Check if the model is available locally or download it."""
        try:
            print(f"Loading model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Model '{model_id}' not found locally or failed to load: {str(e)}")

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
        print(f"Image size: {image.size}")  # Debugging image size

        # Check if the model is available and load it
        model, tokenizer = self.check_model_availability(model_id, revision)

        # Set the pad token if it does not exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Assigned pad_token to eos_token.")

        # Encode the image using the model
        try:
            enc_image = model.encode_image(image)
            if enc_image is None or enc_image.size(0) == 0:
                raise ValueError("Encoded image is empty. Check the input image and model.")
            print(f"Encoded image size: {enc_image.size()}")  # Debugging tensor shape
        except Exception as e:
            raise RuntimeError(f"Error encoding image: {str(e)}")

        # Generate text description from the image
        try:
            description = model.answer_question(enc_image, "Describe this image.", tokenizer)
        except AttributeError as e:
            raise AttributeError(f"Model '{model_id}' does not have 'answer_question' method: {str(e)}")
        except IndexError as e:
            print(f"Error: {str(e)}. This may be due to invalid position embeddings or input size.")
            raise

        if log_prompt == "Yes":
            print(f"ImageToText: {description}")
        
        return [description]

NODE_CLASS_MAPPINGS = {
    "ComfyUI_ImageToText": ComfyUI_ImageToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_ImageToText": "ComfyUI_ImageToText"
}
