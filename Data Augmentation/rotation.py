from PIL import Image
import os

def rotate_images(input_folder, output_folder):
    """
    Rotates all images in the input folder 90 degrees to the left and saves
    them in the output folder.

    Args:
        input_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where the rotated images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(input_path)
                rotated_img = img.rotate(180, expand=True)  # Set rotation here
                output_path = os.path.join(output_folder, f"rotated_{filename}")
                rotated_img.save(output_path)
                print(f"Rotated and saved: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = 'Dataset/Original'  # Replace with the path to your input folder
    output_directory = 'Dataset/Rotated' # Replace with the desired path for the output folder

    # Create the input folder with some dummy images for testing
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        # Create a dummy red image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(os.path.join(input_directory, "red_image.png"))
        # Create a dummy blue image
        img = Image.new('RGB', (50, 80), color='blue')
        img.save(os.path.join(input_directory, "blue_image.jpg"))
        print(f"Created dummy input images in: {input_directory}")

    rotate_images(input_directory, output_directory)
    print("Image rotation complete.")