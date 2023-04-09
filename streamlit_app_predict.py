import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       


       	
	


	

	
def main():
    st.title('Face Similiarity Check')	
   
    image1, svd_img1 = load_image()
    image2, svd_img2 = load_image()
   
    result = st.button('Predict')
    if(result):
        st.image(svd_img1, caption="image1")
        st.image(svd_img2, caption="image2")
	#extractFace(svd_img1)
	#extractFace(svd_img2)
        


    
    

if __name__ == '__main__':
    main()
