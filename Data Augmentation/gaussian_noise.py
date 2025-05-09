import os
from PIL import Image
import numpy as np

def add_gaussian_noise(image, mean=0, stddev=25):
    """Adds Gaussian noise to a PIL Image object.

    Args:
        image (PIL.Image.Image): The input PIL Image.
        mean (float): The mean (average) of the Gaussian distribution.
        stddev (float): The standard deviation of the Gaussian distribution.

    Returns:
        PIL.Image.Image: A new PIL Image with added Gaussian noise.
    """
    img_array = np.array(image, dtype=float)
    noise = np.random.normal(mean, stddev, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def create_noisy_copies(input_folder, output_folder, noise_mean=0, noise_stddev=25):
    """
    Creates noisy copies of all images in the input folder and saves them
    to the output folder.

    Args:
        input_folder (str): The path to the folder containing the original images.
        output_folder (str): The path to the folder where the noisy copies will be saved.
        noise_mean (float): The mean of the Gaussian noise to add.
        noise_stddev (float): The standard deviation of the Gaussian noise to add.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        if is_image_file(filename):
            input_filepath = os.path.join(input_folder, filename)
            output_filename = f"noisy_{filename}"
            output_filepath = os.path.join(output_folder, output_filename)
            try:
                img = Image.open(input_filepath)
                noisy_img = add_gaussian_noise(img, noise_mean, noise_stddev)
                noisy_img.save(output_filepath)
                print(f"Created noisy copy: {filename} -> {output_filename}")
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
    input_folder = '/Dataset/Original'
    output_folder = '/Dataset/Gausian'
    mean_noise = float(input("Enter the mean of the Gaussian noise (default: 0): ") or 0)
    stddev_noise = float(input("Enter the standard deviation of the Gaussian noise (default: 25): ") or 25)

    if os.path.isdir(input_folder):
        create_noisy_copies(input_folder, output_folder, mean_noise, stddev_noise)
        print("Noisy image creation process completed. Noisy copies are saved in:", output_folder)
    else:
        print("Invalid input folder path.")