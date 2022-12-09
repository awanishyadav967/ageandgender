import cv2
import face_detection
import streamlit as st
from PIL import Image
import numpy as np

print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

def crop_object(image, box, num = 0, names  = []):
  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  name = "crop_img_"+str(num)+".png"
  names.append(name)
  crop_img.save(name)
  return crop_img

# define necessary image processing functions

#necessary cuz when we run images through SRCNN based on the kernel sizes and convulational layers, we are going to lose some of these outside pixels
#the images are going to get smaller and that's why it is neccesary to have a divisible image size 
def modcrop(img,scale):
    #temp size
    tmpsz=img.shape
    sz=tmpsz[0:2]
    
    #ensures that dimension of our image are divisible by scale(doesn't leaves hanging remainders) by cropping the images size
    #np.mod returns the remainder bewtween our sz and scale
    sz=sz-np.mod(sz,scale)
    
    img=img[0:sz[0],1:sz[1]]
    return img

#crop offs the bordersize from all sides of the image
def shave(image,border):
    img=image[border: -border,border:-border]
    return img


@st.cache(suppress_st_warning=True)
def faceDetection(input_image_path):
  im = input_image_path[:, :, ::-1]
  detections = detector.detect(im)
  st.write(len(detections))
  #st.write(detections)
  num=0

  image_landmarks = input_image_path
  names = [] 
  boxes = []
  for detections in detections:
    x = int(detections[0])
    y = int(detections[1])
    w = int(detections[2])
    h = int(detections[3])
    cv2.rectangle(image_landmarks, (x, y), (w, h), (0, 255, 0), 2)
    print(x, y, w, h)
    #cv2.putText(image_landmarks, 'X', (w-10, y+10),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255), 3,cv2.LINE_AA )
    boxes.append([x,y,w,h])
    #image = Image.open(input_image_path)
    #st.image(crop_object(image, detections, num, names)
    #crop_object(image, detections, num, names)

    num+=1
  image_landmarks = cv2.cvtColor(image_landmarks, cv2.COLOR_BGR2RGB)
  return image_landmarks, num, boxes