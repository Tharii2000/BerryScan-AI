import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil

# Try to import albumentations, fall back to PIL if it fails
USING_ALBUMENTATIONS = False
try:
    import lzma
    import albumentations as A
    USING_ALBUMENTATIONS = True
except ImportError as e:
    if '_lzma' in str(e):
        print("WARNING: Could not load _lzma DLL needed for albumentations.")
        print("Falling back to PIL-based augmentation.")
        print("\nTo use albumentations in the future, try one of the following solutions:")
        print("1. Install the missing lzma library:")
        print("   conda install -c conda-forge python-lzma")
        print("\n2. Or reinstall your Python environment with all dependencies:")
        print("   conda install -c conda-forge liblzma")
        print("   conda install -c conda-forge xz")
        
        # Import PIL as fallback
        from PIL import Image, ImageEnhance, ImageOps
    else:
        # Re-raise if it's not the lzma issue
        raise

def augment_with_pil(img_path, output_path):
    """Augment an image using PIL transformations"""
    try:
        # Open image with PIL
        image = Image.open(img_path)
        
        # Resize the image to maintain consistency
        image = image.resize((256, 256))
        
        # Apply random transformations
        # Random horizontal flip
        if random.random() > 0.5:
            image = ImageOps.mirror(image)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = ImageOps.flip(image)
        
        # Random rotation
        angle = random.randint(-30, 30)
        image = image.rotate(angle, expand=False)
        
        # Random color jitter
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Save the augmented image
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error augmenting image with PIL: {e}")
        return False

def limit_class_size(class_dir, max_images=700):
    """
    Limit the number of images in a class folder to max_images.
    If there are more images than max_images, randomly delete the excess.
    
    Args:
        class_dir: Directory containing class images
        max_images: Maximum number of images to keep
    """
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    current_count = len(image_files)
    
    if current_count <= max_images:
        return current_count  # No need to delete anything
    
    # Randomly select images to delete
    images_to_remove = random.sample(image_files, current_count - max_images)
    
    print(f"Class has {current_count} images, removing {len(images_to_remove)} to reach target of {max_images}")
    
    # Delete the selected images
    for img_file in tqdm(images_to_remove, desc="Removing excess images"):
        img_path = os.path.join(class_dir, img_file)
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Error removing {img_path}: {e}")
    
    return max_images

def augment_images(input_dir, output_dir, target_count=700):
    """
    Augment images in each class to reach target_count per class.
    If a class has more than target_count images, excess images will be removed.
    
    Args:
        input_dir: Directory containing class folders with images
        output_dir: Directory to save augmented images
        target_count: Target number of images per class
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get class names directly from the folder structure
    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    if USING_ALBUMENTATIONS:
        # Define augmentation transformations with different severities
        light_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8)
        ])
        
        medium_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7),
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomShadow(p=0.2)
        ])
        
        heavy_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomShadow(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
    
    # Process each class folder
    for class_folder in class_folders:
        class_path = os.path.join(input_dir, class_folder)
        output_class_path = os.path.join(output_dir, class_folder)
        
        # Create output class folder if it doesn't exist
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        
        # Get image files
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        current_count = len(image_files)
        
        print(f"\nProcessing class: {class_folder}")
        print(f"Found {current_count} original images")
        
        # Copy original images first
        for img_file in image_files:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(output_class_path, img_file)
            
            # Copy the original image
            if USING_ALBUMENTATIONS:
                img = cv2.imread(src_path)
                if img is not None:
                    cv2.imwrite(dst_path, img)
            else:
                try:
                    img = Image.open(src_path)
                    img.save(dst_path)
                except Exception as e:
                    print(f"Error copying original image {src_path}: {e}")
        
        # Calculate how many more images we need
        needed_count = max(0, target_count - current_count)
        
        if needed_count == 0:
            # If we have enough or too many images
            if current_count > target_count:
                print(f"Class {class_folder} has {current_count} images, limiting to {target_count}")
                # Limit the number of images in the output folder
                limit_class_size(output_class_path, max_images=target_count)
            else:
                print(f"Class {class_folder} already has exactly {current_count} images, no modification needed")
            continue
            
        print(f"Generating {needed_count} additional images through augmentation")
        
        # Strategy: Cycle through original images repeatedly with different augmentations
        augmentations_left = needed_count
        pbar = tqdm(total=needed_count, desc="Generating images")
        
        while augmentations_left > 0:
            for idx, img_file in enumerate(image_files):
                if augmentations_left <= 0:
                    break
                    
                img_path = os.path.join(class_path, img_file)
                filename, ext = os.path.splitext(img_file)
                new_filename = f"{filename}_aug_{idx}_{augmentations_left}{ext}"
                output_path = os.path.join(output_class_path, new_filename)
                
                if USING_ALBUMENTATIONS:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Choose augmentation based on how many more we need to create
                    if current_count < 100:
                        aug_pipeline = random.choice([medium_aug, heavy_aug, heavy_aug])
                    elif current_count < 300:
                        aug_pipeline = random.choice([light_aug, medium_aug, heavy_aug])
                    else:
                        aug_pipeline = random.choice([light_aug, light_aug, medium_aug])
                    
                    # Apply augmentation
                    augmented = aug_pipeline(image=img)
                    aug_img = augmented["image"]
                    cv2.imwrite(output_path, aug_img)
                else:
                    # Use PIL-based augmentation
                    if not augment_with_pil(img_path, output_path):
                        continue
                
                augmentations_left -= 1
                pbar.update(1)
                
                if augmentations_left <= 0:
                    break
        
        pbar.close()
        
        # Count final number of images in output folder
        final_count = len(os.listdir(output_class_path))
        
        # If there are still too many images after augmentation, limit to exactly target_count
        if final_count > target_count:
            print(f"Limiting final count from {final_count} to {target_count}")
            limit_class_size(output_class_path, max_images=target_count)
            final_count = target_count
            
        print(f"Completed class {class_folder}: {final_count} total images")

if __name__ == "__main__":
    # Update these paths to match your directory structure
    input_dir = "c:/Users/Kusal/Desktop/3rd YEAR PROJECTS/Codes/berry/Train"
    output_dir = "c:/Users/Kusal/Desktop/3rd YEAR PROJECTS/Codes/berry/Augmented_Train"
    
    # Augment all classes to have exactly 700 images
    augment_images(input_dir, output_dir, target_count=700)
    print("Image augmentation complete! All classes have exactly 700 images.")