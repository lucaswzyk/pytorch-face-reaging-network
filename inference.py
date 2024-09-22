import torch
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2  # For video reading and writing
from models.generator import Generator
import mediapipe as mp
from PIL import Image

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
generator_model = Generator()
generator_model.load_state_dict(torch.load('models/large-aging-model.h5', map_location=device))
generator_model.to(device)

# MediaPipe setup for segmentation
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='models/selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)
segmenter = ImageSegmenter.create_from_options(options)
COLOR_MAP = {
    0: [0, 0, 0],         # Background (black)
    1: [255, 204, 204],   # Body skin (light red)
    2: [204, 255, 204],   # Face skin (light green)
    3: [204, 204, 255],   # Hair (light blue)
    4: [255, 255, 153],   # Clothing (light yellow)
    5: [255, 153, 255],   # Accessories (light pink)
    6: [153, 255, 255]    # Other (light cyan)
}

# Image transformation
transform_resize = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

# Add a function to save intermediary results for debugging
def save_intermediary_image(image, path):
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise ValueError("Input should be either a numpy ndarray or a PIL Image")

# Function to generate age map based on facial skin mask
def generate_age_map(image, input_age, output_age):
    np_image = np.array(image) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
    segmentation_result = segmenter.segment(mp_image)
    
    # Ensure the category mask has correct shape
    category_mask = segmentation_result.category_mask.numpy_view()
    
    # Initialize the colorized mask
    colorized_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Apply colors from COLOR_MAP to different regions
    for region, color in COLOR_MAP.items():
        colorized_mask[category_mask == region] = color
    
    # Create age_map2 with initial values set to input_age
    age_map2 = np.full((512, 512), input_age / 100)
    
    # Set age_map2 values to output_age only for face and body skin regions (categories 1 and 2)
    age_map2[(category_mask == 2) | (category_mask == 3)] = output_age / 100
    
    # Convert age_map2 to tensor for further use
    age_map2 = torch.tensor(age_map2).unsqueeze(0).to(device)
    
    return age_map2, colorized_mask

# Pic2Pic method for processing images
def process_single_image(image, input_age, output_age, frame_count, intermediary_dir):
    resized_image = image.resize((512, 512))

    input_image = transform_resize(image=np.array(image))['image'] / 255
    age_map1 = torch.full((1, 512, 512), input_age / 100)
    age_map2, colorized_mask = generate_age_map(resized_image, input_age, output_age)

    # Save intermediary images
    save_intermediary_image(colorized_mask, os.path.join(intermediary_dir, f"frame_{frame_count}_segmentation.png"))  # Colorized Segmentation Mask
    save_intermediary_image(age_map2.squeeze(0).cpu().numpy() * 255, os.path.join(intermediary_dir, f"frame_{frame_count}_age_map2.png"))  # Age map2

    input_tensor = torch.cat((input_image, age_map1, age_map2), dim=0)

    # Ensure that all tensors are of the same dtype
    input_tensor = input_tensor.to(torch.float32)  # Convert to float32

    with torch.no_grad():
        model_output = generator_model(input_tensor.unsqueeze(0).to(device))

    processed_output = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
    save_intermediary_image(processed_output * 3, os.path.join(intermediary_dir, f"frame_{frame_count}_2_ai_output.png"))  # Pure AI output
    
    new_image = np.clip((processed_output + np.array(resized_image)), 0, 255).astype('uint8')
    
    np_test = np.array(image)
    sample_image = np.array(Image.fromarray(new_image).resize((np_test.shape[1], np_test.shape[0]))).astype('uint8')

    # Save final result
    save_intermediary_image(sample_image, os.path.join(intermediary_dir, f"frame_{frame_count}_final_result.png"))

    return sample_image

# Crop the center square of the frame
def crop_center_square(frame):
    height, width = frame.shape[:2]
    if width == height:
        return frame  # Already square

    size = min(width, height)
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]

# Resize and process each frame in the video
def process_video(input_video_path, input_age, output_age, output_video_path, intermediary_dir="intermediary_results"):
    if not os.path.exists(intermediary_dir):
        os.makedirs(intermediary_dir)
    
    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    square_size = min(frame_width, frame_height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_size = (512, 512)
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Crop and resize the input frame
        cropped_frame = crop_center_square(frame)
        resized_frame = cv2.resize(cropped_frame, (512, 512))
        save_intermediary_image(resized_frame, os.path.join(intermediary_dir, f"frame_{frame_count}_0_resized.png"))

        # Process the frame through the model
        processed_frame = process_single_image(Image.fromarray(resized_frame), input_age, output_age, frame_count, intermediary_dir)
        save_intermediary_image(processed_frame, os.path.join(intermediary_dir, f"frame_{frame_count}_1_final_result.png"))  

        # Write the processed frame to the output video
        output_video.write(processed_frame)
        
        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}")
    
    video_capture.release()
    output_video.release()
    print(f"Video saved at {output_video_path}")
    print(f"Intermediary results saved at {intermediary_dir}")

# Function to process single image files
def process_image_file(input_image_path, input_age, output_age, output_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    output_image = process_single_image(input_image, input_age, output_age, 0, "intermediary_results")  # Use dummy frame_count for single images
    output_image = Image.fromarray(output_image)
    output_image.save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Main function to handle command-line execution
if __name__ == "__main__":
    input_image_path = "example-images/input_example_img.png"  # Replace with your input image path
    output_image_path = "example-images/output_example_img.png"  # Replace with your output image path
    input_age = 20  # Replace with the current age in the image
    output_age = 80  # Replace with the desired age
    
    if os.path.exists(input_image_path):
        process_image_file(input_image_path, input_age, output_age, output_image_path)
    
    input_video_path = "example-videos/input_example_video.mov"  # Replace with your input video path
    output_video_path = "example-videos/output_example_video.mp4"  # Replace with your output video path
    
    if os.path.exists(input_video_path):
        process_video(input_video_path, input_age, output_age, output_video_path)
  # Replace with your input video path
   
