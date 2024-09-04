from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

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

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text_positive',)
    FUNCTION = "image2text"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI_Mexx"

    def image2text(self, images, log_prompt):
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)
        image = pil_images[0]
        
        # Load the model and tokenizer from Hugging Face
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {str(e)}")
        
        # Assuming the model supports image encoding and text generation
        try:
            enc_image = model.encode_image(image)  # Hypothetical function
            en = model.answer_question(enc_image, "Describe this image.", tokenizer)
        except AttributeError:
            raise AttributeError(f"Model '{model_id}' does not have 'encode_image' or 'answer_question' methods.")
        
        if log_prompt == "Yes":
            print(f"ImageToText: {en}")
        
        return [en]

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComfyUI_ImageToText": ComfyUI_ImageToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_ImageToText": "ComfyUI_ImageToText"
}

