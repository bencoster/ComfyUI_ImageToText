import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the folder containing images
base = './image/'

# Model ID, if switching models, modify this part
model_id = "vikhyatk/moondream2"
revision = "2024-04-02"

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if not f.startswith('.') and not f.endswith('.txt'):
                fullname = os.path.join(root, f)
                yield fullname

def main():
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    for imagefile in findAllFile(base):
        print(f"Processing image: {imagefile}")
        image = Image.open(imagefile)
        enc_image = model.encode_image(image)
        en = model.answer_question(enc_image, "Describe this image.", tokenizer)
        
        file_name, file_extension = os.path.splitext(imagefile)
        print(f"{file_name} natural language tag: {en}")
        
        with open(file_name + ".txt", 'w', encoding='utf-8') as file:
            # Write the description to a file
            file.write(en)
            file.write('\n')

if __name__ == '__main__':
    main()
