import os
from PIL import Image

def flip_images_in_folder(folder_path, output_folder):
    """
    Flips all images (horizontally) in the specified folder and saves
    the flipped images to the output folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        output_folder (str): The path to the folder where the flipped images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            input_filepath = os.path.join(folder_path, filename)
            output_filepath = os.path.join(output_folder, f"flipped_{filename}")
            try:
                img = Image.open(input_filepath)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
                flipped_img.save(output_filepath)
                print(f"Flipped and saved: {filename} -> {os.path.basename(output_filepath)}")
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
    input_folder = input("Enter the path to the folder containing the images: ")
    output_folder = input("Enter the path to the folder where you want to save the flipped images: ")

    if os.path.isdir(input_folder):
        flip_images_in_folder(input_folder, output_folder)
        print("Image flipping process completed. Flipped images are saved in:", output_folder)
    else:
        print("Invalid input folder path.")