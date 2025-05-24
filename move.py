import os
import shutil
import random

def move_images_to_test_set(source_dir='berry/Train', dest_dir='berry/Test', images_per_class=80):
    """
    Move images_per_class images from each class directory in source_dir to dest_dir,
    while maintaining the same folder structure.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('_')]
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    for class_name in class_dirs:
        # Source and destination paths
        class_src_path = os.path.join(source_dir, class_name)
        class_dest_path = os.path.join(dest_dir, class_name)
        
        # Create the class directory in the destination if it doesn't exist
        if not os.path.exists(class_dest_path):
            os.makedirs(class_dest_path)
        
        # Get all image files in the source directory
        all_images = [f for f in os.listdir(class_src_path) if os.path.isfile(os.path.join(class_src_path, f)) 
                      and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        # Randomly select images_per_class images
        if len(all_images) <= images_per_class:
            selected_images = all_images
            print(f"Warning: Only {len(all_images)} images available in {class_name}, moving all")
        else:
            selected_images = random.sample(all_images, images_per_class)
        
        # Move each selected image
        for img_file in selected_images:
            src_file = os.path.join(class_src_path, img_file)
            dst_file = os.path.join(class_dest_path, img_file)
            
            # Copy the file to destination
            shutil.copy2(src_file, dst_file)
            
            # Remove the original file (for move operation)
            os.remove(src_file)
            
        print(f"Moved {len(selected_images)} images from {class_name}")

# Example usage
if __name__ == "__main__":
    # Update these paths to match your actual directory structure
    source_dir = 'c:/Users/Kusal/Desktop/3rd YEAR PROJECTS/Codes/berry/Train'
    dest_dir = 'c:/Users/Kusal/Desktop/3rd YEAR PROJECTS/Codes/berry/Test'
    
    # Move 80 images per class
    move_images_to_test_set(source_dir, dest_dir, 80)
    print("Done! Images have been moved from Train to Test folders.")