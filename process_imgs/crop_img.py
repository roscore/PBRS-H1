

# Importing Image class from PIL module
from PIL import Image



def crop_img(input_img_path):
 
    # Opens a image in RGB mode
    im = Image.open(input_img_path)
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    width_base = width / 4
    height_base = height / 4
    
    # Setting the points for cropped image
    left = width / 4
    top = height_base * 1
    right = 3 * width / 4  
    bottom = top + 2 * height_base
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))

    return im1

if __name__ == "__main__":
    input_img_path = "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/imgs/h1_v84/000.png"
    output_img_path = "test.png"

    im1 = crop_img(input_img_path)

    print(str(im1.size))

    im1.save(output_img_path)
 
# # Shows the image in image viewer
# im1.show()

