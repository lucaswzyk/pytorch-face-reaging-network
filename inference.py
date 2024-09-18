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

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model classes
class BlurUpSample(nn.Module):
    def __init__(self, c):
        super(BlurUpSample, self).__init__()
        self.blurpool = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.blurpool(x)
        x = self.upsample(x)
        return x

class DownLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownLayer, self).__init__()
        self.maxblurpool = kornia.filters.MaxBlurPool2D(kernel_size=3)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.maxblurpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        return x

class UpLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(UpLayer, self).__init__()
        self.upsample = BlurUpSample(c_in)
        self.conv1 = nn.Conv2d(c_in + c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x, skip_x):
        x = self.upsample(x)
        dh = skip_x.size(2) - x.size(2)
        dw = skip_x.size(3) - x.size(3)
        x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downlayer1 = DownLayer(64, 128)
        self.downlayer2 = DownLayer(128, 256)
        self.downlayer3 = DownLayer(256, 512)
        self.downlayer4 = DownLayer(512, 1024)
        self.uplayer1 = UpLayer(1024, 512)
        self.uplayer2 = UpLayer(512, 256)
        self.uplayer3 = UpLayer(256, 128)
        self.uplayer4 = UpLayer(128, 64)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.batchnorm1(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.conv2(x1)
        x1 = self.batchnorm1(x1)
        x1 = self.leakyrelu(x1)
        x2 = self.downlayer1(x1)
        x3 = self.downlayer2(x2)
        x4 = self.downlayer3(x3)
        x5 = self.downlayer4(x4)
        x = self.uplayer1(x5, x4)
        x = self.uplayer2(x, x3)
        x = self.uplayer3(x, x2)
        x = self.uplayer4(x, x1)
        x = self.conv3(x)
        return x

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
