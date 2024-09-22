import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from models.generator import Generator

# Constants
COLOR_MAP = {
    0: [0, 0, 0],         # Background (black)
    1: [255, 204, 204],   # Body skin (light red)
    2: [204, 255, 204],   # Face skin (light green)
    3: [204, 204, 255],   # Hair (light blue)
    4: [255, 255, 153],   # Clothing (light yellow)
    5: [255, 153, 255],   # Accessories (light pink)
    6: [153, 255, 255]    # Other (light cyan)
}

# Image transformation configuration
IMAGE_TRANSFORM = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

# MediaPipe setup for segmentation
def create_segmenter():
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path='models/selfie_multiclass_256x256.tflite'),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True
    )
    return ImageSegmenter.create_from_options(options)

segmenter = create_segmenter()

# Load model
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator()
    model.load_state_dict(torch.load('models/large-aging-model.h5', map_location=device))
    model.to(device)
    return model, device

generator_model, device = load_model()

# Utility to save intermediary images
def save_image(image, path):
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise ValueError("Input should be either a numpy ndarray or a PIL Image")

# Function to generate age map based on face and body segmentation
def generate_age_map(image, input_age, output_age):
    np_image = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
    segmentation_result = segmenter.segment(mp_image)
    
    category_mask = segmentation_result.category_mask.numpy_view()
    colorized_mask = np.zeros((512, 512, 3), dtype=np.uint8)

    for region, color in COLOR_MAP.items():
        colorized_mask[category_mask == region] = color

    age_map = np.full((512, 512), input_age / 100)
    age_map[(category_mask == 2) | (category_mask == 3)] = output_age / 100

    age_map_tensor = torch.tensor(age_map).unsqueeze(0).to(device)
    return age_map_tensor, colorized_mask

# Process single image (Pic2Pic)
def process_image(image, input_age, output_age, frame_idx, intermediary_dir):
    resized_image = image.resize((512, 512))
    input_image = IMAGE_TRANSFORM(image=np.array(image))['image'] / 255

    age_map1 = torch.full((1, 512, 512), input_age / 100)
    age_map2, colorized_mask = generate_age_map(resized_image, input_age, output_age)

    save_image(colorized_mask, os.path.join(intermediary_dir, f"frame_{frame_idx}_segmentation.png"))
    save_image(age_map2.squeeze(0).cpu().numpy() * 255, os.path.join(intermediary_dir, f"frame_{frame_idx}_age_map2.png"))

    input_tensor = torch.cat((input_image, age_map1, age_map2), dim=0).float().to(device)

    with torch.no_grad():
        model_output = generator_model(input_tensor.unsqueeze(0))

    processed_output = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
    save_image(processed_output * 3, os.path.join(intermediary_dir, f"frame_{frame_idx}_ai_output.png"))

    blended_image = np.clip((processed_output + np.array(resized_image)), 0, 255).astype('uint8')
    final_image = np.array(Image.fromarray(blended_image).resize(image.size))

    save_image(final_image, os.path.join(intermediary_dir, f"frame_{frame_idx}_final_result.png"))
    return final_image

# Crop the center square from a frame
def crop_center_square(frame):
    height, width = frame.shape[:2]
    size = min(width, height)
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]

# Process video frame by frame
def process_video(input_video_path, input_age, output_age, output_video_path, intermediary_dir="intermediary_results"):
    if not os.path.exists(intermediary_dir):
        os.makedirs(intermediary_dir)

    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))

    for idx in range(frame_count):
        ret, frame = video_capture.read()
        if not ret:
            break

        cropped_frame = crop_center_square(frame)
        resized_frame = cv2.resize(cropped_frame, (512, 512))
        save_image(resized_frame, os.path.join(intermediary_dir, f"frame_{idx}_resized.png"))

        processed_frame = process_image(Image.fromarray(resized_frame), input_age, output_age, idx, intermediary_dir)
        save_image(processed_frame, os.path.join(intermediary_dir, f"frame_{idx}_final_result.png"))

        output_video.write(processed_frame)
        print(f"Processed frame {idx + 1}/{frame_count}")

    video_capture.release()
    output_video.release()
    print(f"Video saved at {output_video_path}")
    print(f"Intermediary results saved at {intermediary_dir}")

# Process a single image file
def process_image_file(input_image_path, input_age, output_age, output_image_path):
    image = Image.open(input_image_path).convert('RGB')
    final_image = process_image(image, input_age, output_age, 0, "intermediary_results")
    Image.fromarray(final_image).save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Main function to run from command-line
if __name__ == "__main__":
    input_image_path = "example-images/input_example_img.png"
    output_image_path = "example-images/output_example_img.png"
    input_age = 20
    output_age = 80

    if os.path.exists(input_image_path):
        process_image_file(input_image_path, input_age, output_age, output_image_path)

    # input_video_path = "example-videos/input_example_video.mov"
    # output_video_path = "example-videos/output_example_video.mp4"

    # if os.path.exists(input_video_path):
    #     process_video(input_video_path, input_age, output_age, output_video_path)

    input_video_path = "example-videos/input_example_warum.mov"
    output_video_path = "example-videos/output_example_warum.mp4"

    if os.path.exists(input_video_path):
        process_video(input_video_path, input_age, output_age, output_video_path)
