import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from models.generator import Generator
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor
from math import ceil

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

sr_model = None
square_size = -1

# Function to initialize processing
def setup_processing(image):
    global sr_model, square_size

    print("Setting up processing...")
    # Get dimensions
    width, height = image.size
    print("(width, height):", (width, height))
    # Determine square crop size
    square_size = min(width, height)
    print("Square size:", square_size)
    # Determine the scale factor
    sr_factor = ceil(square_size / 512)
    print("SR Factor:", sr_factor)
    
    # Load appropriate SR model
    if sr_factor > 1: sr_model = ninasr_b0(scale=sr_factor, pretrained=True)



# Function to perform scaling (upscaling or downscaling)
def scale_square(image, size):
    if image.size[0] < size and sr_model is not None:
        save_intermediate_image(image, "upscaled_pre.jpg")
        # Upscaling: Super resolution followed by downscaling
        image = np.array(image) / 255.0  # Scale to [0, 1]
        lr_t = to_tensor(image).unsqueeze(0).float()  # Ensure it's float32
        sr_t = sr_model(lr_t)
        image = to_pil_image(sr_t.squeeze(0).clamp(0, 1))

    image = image.resize((size, size))
    save_intermediate_image(image, "upscaled_post.jpg")
    
    return image

# Crop the center square from a frame
def crop_center_square(frame):
    height, width = frame.shape[:2]
    size = min(width, height)
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]

def add_center_back_on(original_frame, processed_square, feather_radius=10, corner_radius=50, margin = 20):
    """
    Adds the processed square image back onto the original frame with rounded and feathered edges.
    
    Args:
        original_frame (np.array): The original frame where the square image was cropped from.
        processed_square_image (PIL.Image.Image): The processed square image to add back onto the original frame.
        feather_radius (int): The radius for feathering the edges. Default is 50.
        corner_radius (int): The radius of the rounded corners. Default is 50.
    
    Returns:
        np.array: The original frame with the processed square image added back.
    """
    width, height = original_frame.size
    square_x = (width - square_size) // 2
    square_y = (height - square_size) // 2

    # Create a mask with rounded corners
    mask = Image.new('L', (square_size, square_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([margin, margin, square_size - margin, square_size - margin], radius=corner_radius, fill=255)
    
    # Feather the edges of the mask
    mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))
    # mask.save(os.path.join(intermediate_dir, "mask.jpg"), quality=200)
    
    # Add the square image with feathered mask back onto the original frame
    original_frame.paste(processed_square, (square_x, square_y), mask)
    
    return original_frame

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
def process_image(image, input_age, output_age, frame_idx):
    # Crop and resize the image
    cropped_image = crop_center_square(np.array(image))
    resized_image = scale_square(Image.fromarray(cropped_image), 512)

    # save_intermediate_image(resized_image, f"frame_{frame_idx}_resized.png")

    input_image = IMAGE_TRANSFORM(image=np.array(resized_image))['image'] / 255

    age_map1 = torch.full((1, 512, 512), input_age / 100)
    age_map2, colorized_mask = generate_age_map(resized_image, input_age, output_age)

    save_intermediate_image(colorized_mask, f"frame_{frame_idx}_segmentation.png")
    # save_intermediate_image(age_map2.squeeze(0).cpu().numpy() * 255, f"frame_{frame_idx}_age_map2.png")

    input_tensor = torch.cat((input_image, age_map1, age_map2), dim=0).float().to(device)

    with torch.no_grad():
        model_output = generator_model(input_tensor.unsqueeze(0))

    processed_output = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
    # save_intermediate_image(processed_output * 3, f"frame_{frame_idx}_ai_output.png")

    blended_image = np.clip((processed_output + np.array(resized_image)), 0, 255).astype('uint8')
    save_intermediate_image(blended_image, f"frame_{frame_idx}_blended.png")
    final_square = scale_square(Image.fromarray(blended_image), square_size)

    # save_intermediate_image(final_square, f"frame_{frame_idx}_final_square.png")

    return add_center_back_on(image, final_square)

intermediate_dir="intermediate_results"
# Utility to save intermediary images
def save_intermediate_image(image, path):
    path = os.path.join(intermediate_dir, path)
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise ValueError("Input should be either a numpy ndarray or a PIL Image")

# Process video frame by frame
def process_video(input_video_path, input_age, output_age, output_video_path):
    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


    for idx in range(frame_count):
        ret, frame = video_capture.read()
        if not ret:
            print(f"Reached end of video at frame {idx}")
            break

        image = Image.fromarray(frame)
        if not idx: 
            setup_processing(image)
            output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, image.size)

        processed_frame = process_image(image, input_age, output_age, idx)
        save_intermediate_image(processed_frame, f"frame_{idx}_final_result.png")

        output_video.write(np.array(processed_frame))
        print(f"Processed frame {idx + 1}/{frame_count}")

    video_capture.release()
    output_video.release()
    print(f"Video saved at {output_video_path}")
    print(f"Intermediary results saved at {intermediate_dir}")

# Process a single image file
def process_image_file(input_image_path, input_age, output_age, output_image_path):
    image = Image.open(input_image_path).convert('RGB')
    setup_processing(image)
    final_image = process_image(image, input_age, output_age, 0)
    final_image.save(output_image_path)
    print(f"Image saved at {output_image_path}")

# Main function to run from command-line
if __name__ == "__main__":
    input_image_path = "example-images/input_example_img.png"
    output_image_path = "example-images/output_example_img.png"
    input_age = 20
    output_age = 80

    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    if os.path.exists(input_image_path):
        process_image_file(input_image_path, input_age, output_age, output_image_path)

    video_path_stub = "example-videos/founder_short_big"
    input_video_path = video_path_stub + ".mov"
    output_video_path = video_path_stub + "_out.mp4"
    # input_video_path = "example-videos/input_example_video.mov"

    if os.path.exists(input_video_path):
        process_video(input_video_path, input_age, output_age, output_video_path)
