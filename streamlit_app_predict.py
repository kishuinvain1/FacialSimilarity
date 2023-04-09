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
    img_list = []
    uploaded_files = st.file_uploader(label='Upload images to find the similarity', accept_multiple_files=True)
    for uploaded_file in uploaded_files: 
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            image_data = uploaded_file.getvalue() 
            #st.image(image_data)
            name = uploaded_file.name
            path = os.path.abspath(name)
            print("abs path")
            print(path)
            img_list.append(opencv_image)
    return img_list



	


	

	
def main():
    st.title('Face Similiarity Check')	
   
    svd_img_list = load_image()
    
   
    result = st.button('Predict')
    if(result):
        for image in svd_img_list:
            st.image(image, caption="image1")
       
	    #extractFace(svd_img1)

        


    
    

if __name__ == '__main__':
    main()
