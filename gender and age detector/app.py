from io import BufferedReader, BytesIO

import streamlit as st
from PIL import Image

import json
import cv2
import numpy as np
import os

os.system("pip list")
os.system("/home/appuser/venv/bin/python -m pip install basicsr")

from srgan import predictSrgan

from face_dectec import crop_object, faceDetection
from srcnn import predictCNN
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from helper_functions import *
from numpy import asarray

import zipfile 

#os.system("/home/appuser/venv/bin/python -m pip install --upgrade pip")
# Page config
#st.set_page_config(page_title="SuperResolution",layout="wide")
# app design
icon = Image.open('extra/icon2.ico')
app_meta(icon)


set_bg_hack('extra/bq4.png')



#style
styl = f"""
<style>
  .st-bg {{
    background-color: rgb(207 226 224 / 77%);
  }}
	.css-zn0oak{{
    background-color: rgb(207 226 224);
    padding: 3rem 1rem;
	}}
  .css-15euf4{{
    background-color: rgb(207 226 224);
    padding: 3rem 1rem;
	}}
  h6{{
    color:rgb(0 104 201 / 75%);
	}}
  .css-83m1w6 a {{
    color:rgb(0 104 201 / 75%);
  }}
	}}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


# set logo in sidebar using PIL
logo = Image.open('extra/name.png')
#st.sidebar.image(logo,use_column_width=True)
col1, col2, col3 = st.columns([1,2,1])
with col1:st.write(' ')
with col2:st.image('extra/icon2.png')
with col3:st.write(' ')


# Main panel setup
#display_app_header(main_txt='Super Resolution', sub_txt='Upload, procces, download to get a new resolution')

# create ss object
if "expandedval" not in st.session_state:
  st.session_state.expandedval = True

check =False
# Info
with st.expander("What is this app?", expanded=st.session_state.expandedval):    
    st.write("""
            This web-based application allows you to identify faces, resize and download images in just a few clicks.
            All you have to do is to upload a single photo, and follow the guidelines in the sidebar.\n
            This app uses Deep Learning (DL) to:
            * __Identify faces__: It returns the croped image (you can change it).
            * __Increase face resolution__: It returns the image whith a x2 scale.
            \n
            """)
    #test
    coltry1, coltry2, coltry3 = st.columns(3)
    colbtt1, colbtt2, colbtt3 = st.columns([4,1,4])

    with coltry1:st.write(' ')
    with coltry2:
      display_app_header(main_txt = "WANNA TRY?")
      st.image('extra/selfie3.jpeg')
    with coltry3:st.write(' ')

    with colbtt1:st.write(' ')
    with colbtt2: 
      check = st.checkbox("YES", value=check)
    with colbtt3:st.write(' ')

#st.markdown("""---""")
    



#sidebar
st.sidebar.image('extra/upload.png', use_column_width=True)
#st.sidebar.app_section_button("[GitHub](https://github.com/angelicaba23/app-super-resolution)")

display_app_header(main_txt = "📤 Step 1",
                  #sub_txt= "Upload data",
                  is_sidebar=True)

image_file = st.sidebar.file_uploader("Upload Image", type=["png","jpg","jpeg"]) #<class 'streamlit.uploaded_file_manager.UploadedFile'>

display_mini_text("By uploading an image or URL you agree to our ","https://github.com/angelicaba23/app-super-resolution/blob/dev/extra/termsofservice.md","Terms of Service",is_sidebar = True)

if image_file is not None or check:
  if check:
    image_file = 'extra/selfie3.jpeg'
    opencv_image= cv2.imread('extra/selfie3.jpeg')
  #save_image(image_file, image_file.name)
  #img_file = "uploaded_image/" + image_file.name

  else:
    check = False
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8) #<class 'numpy.ndarray'>
    opencv_image = cv2.imdecode(file_bytes, 1) #<class 'numpy.ndarray'>
    

  st.session_state.expandedval = False
  #check = False
  [img_faces, num, boxes] = faceDetection(opencv_image)
  print("numero de rostros = "+ str(num))
  #st.write(boxes)
  #st.image(img_faces)

  display_app_header(main_txt = "🛠️ Step 2",
                sub_txt= "Edit Image",
                is_sidebar=True)
  list = []
  filename = 'saved_state.json'

  for boxes in boxes:
    list.append({
      "type": "rect",
        "left": boxes[0],
        "top": boxes[1],
        "width": boxes[2]-boxes[0],
        "height": boxes[3]-boxes[1],
        "fill": "#00FFB350",
        "stroke": "#00FFB3",
        "strokeWidth": 3
    })

  # Verify updated list
  #st.write(list)

  listObj = {
      "version": "4.4.0",
      "objects": list  
  }

  # Verify updated listObj
  #st.write(listObj)

  with open(filename, 'w') as json_file:
    json.dump(listObj, json_file, 
                        indent=4,  
                        separators=(',',': '))

  with open(filename, "r") as f:   saved_state = json.load(f)
  #st.write(saved_state)
  
  bg_image = Image.open(image_file)
  #image_file = None
  label_color = (
      st.sidebar.color_picker("Annotation color: ", "#97fdf5") 
  )  # for alpha from 00 to FF

  tool_mode = st.sidebar.selectbox(
    "Select faces tool:", ("Add", "Move & edit")
  )
  mode = "transform" if tool_mode=="Move & edit" else "rect"

  canvas_result = st_canvas(
      fill_color=label_color+ "50",
      stroke_width=3,
      stroke_color=label_color,
      background_image=bg_image,
      height=bg_image.height,
      width=bg_image.width,
      initial_drawing=saved_state,
      drawing_mode=mode,
      key="canvas_"+str(bg_image.height)+str(bg_image.width),
  )

  print("Canvas creado")
  if canvas_result.json_data is not None:
    rst_objects = canvas_result.json_data["objects"]
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    n = int(len(rst_objects))

    cols = st.columns(n)
    st.info("🪄 IMAGE PROCESSED BY THE METHOD SUPER RESOLUTION (CNN)")
    cols_srcnn = st.columns(n)
    st.success("🪄 IMAGE PROCESSED BY THE METHOD SUPER RESOLUTION + ENHANCEMENT (GAN)")
    cols_srgan = st.columns(n)
    i = 0

    zf = zipfile.ZipFile('imgs.zip', 'w')

    folders = [
        "gan/",
        "cnn/",
        ]

    for n in folders:
        zfi = zipfile.ZipInfo(n)
        zf.writestr(zfi, '')

    for rst_objects in rst_objects:
      rts_boxes = [rst_objects['left'],rst_objects['top'],rst_objects['width']+rst_objects['left'],rst_objects['height']+rst_objects['top']]
      #st.write(rts_boxes)
      crop_image = crop_object(bg_image, rts_boxes)
      cols[i].image(crop_image)

      #-------CNN-----
      im_bgr = predictCNN(crop_image)

      cols_srcnn[i].image(im_bgr)

      im_rgb = im_bgr[:, :, [2, 1, 0]] #numpy.ndarray
      cv2.imwrite("results/restored_imgs/crop_img_0.png", im_rgb)
      zf.write("results/restored_imgs/crop_img_0.png", f'cnn/crop_img_{str(i)}.png')
      ret, img_enco = cv2.imencode(".png", im_rgb)  #numpy.ndarray
      srt_enco = img_enco.tobytes()   #bytes
      img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
      img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader

      cols_srcnn[i].download_button(
        label="📥",
        data=img_BufferedReader,
        file_name="srcnn_img_"+str(i)+".png",
        mime="image/png"
      )

      #cols_srgan[i].image(predictSrgan(crop_image))
      #cols_srgan[i].image(predictSrgan("crop_img_0.png"))
      img_gan=predictSrgan("crop_img_0.png")
      #img_gan = im_bgr
      cols_srgan[i].image(img_gan)
      with open("results/restored_imgs/crop_img_0.png", "rb") as file:
        cols_srgan[i].download_button(
        label="📥",
        data=file,
        file_name="srgan_img_"+str(i)+".png",
        mime="image/png"
        )
      # Add multiple files to the zip
      zf.write('results/restored_imgs/crop_img_0.png', f'gan/crop_img_{str(i)}.png')
      
      
      print("img" + str(i))
      i += 1
      

    # close the Zip File
    zf.close()

    display_app_header(main_txt = "🎉 Step 3",
          sub_txt= "Download results",
          is_sidebar=True)
    with open("imgs.zip", "rb") as fp:
      btn = st.sidebar.download_button(
          label="📥",
          data=fp,
          file_name="imgs.zip",
          mime="application/zip"
      )
   
  else:
    st.warning("Please select the face manually using the tools.")
      
 #else:
  #  st.error("We have not detected faces in the image, please select the area manually using the tools.")
      
else:
  st.error("Please upload a image to process")
    