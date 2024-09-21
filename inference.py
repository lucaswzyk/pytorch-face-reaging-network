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
import cv2  # For handling video input/output
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

# Function to apply the age filter to an image
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

# Function to process a single image file
def process_image(input_image_path, input_age, output_age, output_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    output_image = age_filter(input_image, input_age, output_age)
    output_image = Image.fromarray(output_image)
    output_image.save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Function to process video file frame by frame
def process_video(input_video_path, output_video_path, input_age, output_age):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return

    # Get the video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use different codecs
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video... Total frames: {frame_count}")
    
    # Process each frame
    for i in range(frame_count):
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Convert frame (numpy array) to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply age filter to the frame
        processed_frame = age_filter(pil_image, input_age, output_age)
        
        # Convert processed frame (PIL) back to numpy array for video writing
        processed_frame_np = np.array(processed_frame)
        processed_frame_np = cv2.cvtColor(processed_frame_np, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        # Write the processed frame to the output video
        video_writer.write(processed_frame_np)
        
        print(f"Processed {i}/{frame_count} frames")

    # Release everything
    video_capture.release()
    video_writer.release()
    print(f"Video saved at {output_video_path}")

# Main function to execute the script
if __name__ == "__main__":
    # Test image processing
    input_image_path = "example-images/input_example_img.png"  # Replace with your input image path
    output_image_path = "example-images/output_example_img.png"  # Replace with your output image path
    input_age = 20  # Replace with the current age in the image
    output_age = 80  # Replace with the desired age
    
    # # Process single image
    # if os.path.exists(input_image_path):
    #     process_image(input_image_path, input_age, output_age, output_image_path)
    # else:
    #     print(f"Input image not found at {input_image_path}")

    # Test video processing
    input_video_path = "example-videos/input_example_video_short.mov"  # Replace with your input video path
    output_video_path = "example-videos/output_example_video80.mp4"  # Replace with your output video path

    # Process video if the file exists
    if os.path.exists(input_video_path):
        process_video(input_video_path, output_video_path, input_age, output_age)
    else:
        print(f"Input video not found at {input_video_path}")
