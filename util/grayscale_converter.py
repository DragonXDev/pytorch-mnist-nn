from PIL import Image

# Convert an image to grayscale
def convert_to_grayscale(image_path, output_path):
    img = Image.open(image_path)
    gray_img = img.convert("L")
    gray_img.save(output_path)

if __name__ == "__main__":
    convert_to_grayscale("img.png", "GS.png")
