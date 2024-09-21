# Imports remain the same
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
import cv2  # For video reading and writing
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

# Pic2Pic method for processing images
def process_single_image(image, input_age, output_age):
    resized_image = image.resize((512, 512))
    
    input_image = transform_resize(image=np.array(image))['image'] / 255
    age_map1 = torch.full((1, 512, 512), input_age / 100)
    age_map2 = torch.full((1, 512, 512), output_age / 100)
    input_tensor = torch.cat((input_image, age_map1, age_map2), dim=0)

    with torch.no_grad():
        model_output = generator_model(input_tensor.unsqueeze(0).to(device))

    # Convert model output to numpy and multiply by 255
    processed_output = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255

    # Add the processed output to the resized image, ensuring no overflow or underflow
    new_image = np.clip((processed_output + resized_image), 0, 255).astype('uint8')
    
    # Resize to original image dimensions
    np_test = np.array(image)
    sample_image = np.array(Image.fromarray(new_image).resize((np_test.shape[1], np_test.shape[0]))).astype('uint8')
    
    return sample_image

# Crop the center square of the frame
def crop_center_square(frame):
    height, width = frame.shape[:2]
    if width == height:
        return frame  # Already square

    # Determine the size of the largest square
    size = min(width, height)

    # Calculate the center crop coordinates
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    cropped_frame = frame[y_start:y_start + size, x_start:x_start + size]
    
    return cropped_frame

# Add a function to save intermediary results for debugging
def save_intermediary_image(image, path):
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    elif isinstance(image, Image.Image):
        image.save(path)

# Resize and process each frame in the video
def process_video(input_video_path, input_age, output_age, output_video_path, intermediary_dir="intermediary_results"):
    # Create a folder for intermediary results if it doesn't exist
    if not os.path.exists(intermediary_dir):
        os.makedirs(intermediary_dir)
    
    # Open the input video
    video_capture = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count
    
    # Calculate the size of the largest square from the center of the frames
    square_size = min(frame_width, frame_height)
    
    # Open a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 output
    output_size = (512, 512)  # Output video size (512x512 for model input)
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit when no more frames are available

        # Save the original frame for debugging
        save_intermediary_image(frame, os.path.join(intermediary_dir, f"frame_{frame_count}_original.png"))

        # Crop the center square from the frame
        cropped_frame = crop_center_square(frame)
        save_intermediary_image(cropped_frame, os.path.join(intermediary_dir, f"frame_{frame_count}_cropped.png"))

        # Resize to 512x512
        resized_frame = cv2.resize(cropped_frame, (512, 512))
        save_intermediary_image(resized_frame, os.path.join(intermediary_dir, f"frame_{frame_count}_resized.png"))
        
        # Convert to PIL Image for processing
        # pil_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        
        # Process the frame through the model
        processed_frame = process_single_image(Image.fromarray(resized_frame), input_age, output_age)
        save_intermediary_image(processed_frame, os.path.join(intermediary_dir, f"frame_{frame_count}_processed.png"))
        
        # Convert back to OpenCV format (BGR for video)
        # processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Write the processed frame to the output video
        output_video.write(processed_frame)
        
        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}")
    
    # Release resources
    video_capture.release()
    output_video.release()
    print(f"Video saved at {output_video_path}")
    print(f"Intermediary results saved at {intermediary_dir}")

# Function to process single image files
def process_image_file(input_image_path, input_age, output_age, output_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    output_image = process_single_image(input_image, input_age, output_age)
    output_image = Image.fromarray(output_image)
    output_image.save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Main function to handle command-line execution
if __name__ == "__main__":
    # Example for image processing
    input_image_path = "example-images/input_example_img.png"  # Replace with your input image path
    output_image_path = "example-images/output_example_img.png"  # Replace with your output image path
    input_age = 20  # Replace with the current age in the image
    output_age = 40  # Replace with the desired age
    
    # Process a single image
    if os.path.exists(input_image_path):
        process_image_file(input_image_path, input_age, output_age, output_image_path)
    
    # Example for video processing
    input_video_path = "example-videos/input_example_video.mov"  # Replace with your input video path
    output_video_path = "example-videos/output_example_video.mp4"  # Replace with your output video path
    
    if os.path.exists(input_video_path):
        process_video(input_video_path, input_age, output_age, output_video_path)
