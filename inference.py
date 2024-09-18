import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from models.generator import Generator 

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
generator_model = Generator()
generator_model.load_state_dict(torch.load('models/large-aging-model.h5', map_location=device))
generator_model.to(device)

# Image transformation
transform_resize = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

# Function to apply the age filter
def age_filter(image, input_age, output_age):
    resized_image = image.resize((512, 512))
    input_image = transform_resize(image=np.array(image))['image'] / 255
    age_map1 = torch.full((1, 512, 512), input_age / 100)
    age_map2 = torch.full((1, 512, 512), output_age / 100)
    input_tensor = torch.cat((input_image, age_map1, age_map2), dim=0)

    with torch.no_grad():
        model_output = generator_model(input_tensor.unsqueeze(0).to(device))

    np_test = np.array(image)
    new_image = (model_output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255 + np.array(resized_image)).astype('uint8')
    sample_image = np.array(Image.fromarray(new_image).resize((np_test.shape[1], np_test.shape[0]))).astype('uint8')
    return sample_image

# Function to process the image
def process_image(input_image_path, input_age, output_age, output_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    output_image = age_filter(input_image, input_age, output_age)
    output_image = Image.fromarray(output_image)
    output_image.save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Main function to execute the script
if __name__ == "__main__":
    input_image_path = "example-images/input_example_img.png"  # Replace with your input image path
    output_image_path = "example-images/output_example_img.png"  # Replace with your output image path
    input_age = 20  # Replace with the current age in the image
    output_age = 40  # Replace with the desired age
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Input image not found at {input_image_path}")
    else:
        process_image(input_image_path, input_age, output_age, output_image_path)
