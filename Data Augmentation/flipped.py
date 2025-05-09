import os
from PIL import Image

def flip_images_in_folder(folder_path):
    """
    Flips all images (horizontally) in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the images.
    """
    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
                flipped_img.save(filepath)  # Overwrite the original image
                print(f"Flipped: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def is_image_file(filename):
    """
    Checks if a filename has a common image extension.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the filename ends with a common image extension, False otherwise.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing the images: ")
    if os.path.isdir(folder_path):
        flip_images_in_folder(folder_path)
        print("Image flipping process completed.")
    else:
        print("Invalid folder path.")