import cv2
import face_recognition
from google.colab.patches import cv2_imshow
import glob
import gradio as gr

def act(img):
  # image = cv2.imread(img)
  rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  encode_img = face_recognition.face_encodings(rgb)[0]

  imdir = '/content/' # add custom path name to traves and compare images in directory
  ext = ['png', 'jpg', 'gif','jpeg']    # Add image formats here

  files = []
  [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

  images = [cv2.imread(file) for file in files]
  y=0
  for i in images:
    rgb1= cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
    encode_img1 = face_recognition.face_encodings(rgb1)[0]
    out = face_recognition.compare_faces([encode_img],encode_img1)
    if out==True:
      break
    y+=1
  return str(y)

def snap(img):
  return act(img)
  

g=gr.Interface(snap, gr.inputs.Image(source="webcam", tool=None), "text")
g.launch(debug=True,)
