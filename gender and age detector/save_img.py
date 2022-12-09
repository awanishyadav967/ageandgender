import os

 # @ save image
def save_image(image_file, image_name):
  with open(os.path.join("uploaded_image", image_name), "wb") as f:
    f.write(image_file.getbuffer())