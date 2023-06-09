import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64

from keras.models import load_model
import keras_facenet

import string
import random


def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]





def load_image():
    opencv_image = None 
    path = None
    f = None
    img_list = []
    img_nms = []
    uploaded_files = st.file_uploader(label='Upload images to find the similarity', accept_multiple_files=True)
    for uploaded_file in uploaded_files: 
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            image_data = uploaded_file.getvalue() 
            #st.image(image_data)
            name = uploaded_file.name
            img_nms.append(name)
            path = os.path.abspath(name)
            print("abs path")
            print(path)
            img_list.append(opencv_image)
    return img_list, img_nms



	


def extractFace(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)
    print("[INFO] Found {0} Faces.".format(len(faces)))
#saving every face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
       
    return roi_color
    

@st.cache_resource(ttl=86400)
def load_model():
    embedder = keras_facenet.FaceNet()
    
    return embedder
   
    

def simMeasure(embedding_lst, option, ind):
    dist_list = []
    pnt1 = embedding_lst[ind]
    for embed in embedding_lst:
        dist = np.linalg.norm(pnt1 - embed)
        dist_list.append(dist)

    return dist_list
            
        

	
def main():
    
    st.title('Face Similiarity Check')	
   
    svd_img_list, svd_nms_list = load_image()
    option = st.selectbox('Select Source Image', (svd_nms_list))
   
   
   
   
    result = st.button('Predict')
    embedder = load_model()
    
    embedding_lst = []
    roi_list = []
    if(result):
        st.balloons()
        for image in svd_img_list:
            #st.image(image, caption="image")
            #roi = extractFace(image)
            embedder.name =  ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=7))
           
            face = embedder.extract(image, threshold=0.6)
            box = face[0]['box']
            print("The box is...########################################################################")
            print(box)
            roi = image[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2]),  :]
            exp_roi = np.expand_dims(roi, axis=0)
            print("shape after expansions..........................")
            print(exp_roi.shape)
            print(roi.shape)
	
            #print(roi)

            #st.image(roi, caption="face")
            roi_list.append(roi)
            embedding = embedder.embeddings(exp_roi)
	    #embedding = get_embedding(model, roi)
            embedding_lst.append(embedding)
		
            #image = cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 255, 255), 5 )
            #st.image(image, caption="rect")
		
           
            #simMeasure(embedding_lst, )

        
        ind = svd_nms_list.index(option)
        print("index issssssssssssssss............")
        print(ind)
        final_result = simMeasure(embedding_lst, option, ind)
        print("final_result is.....................")
        print(final_result)
        for i, item in enumerate(final_result):
            img1 = roi_list[ind]
            img2 = roi_list[i]
            img1 = cv2.resize(img1, (100, 100), interpolation = cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (100, 100), interpolation = cv2.INTER_LINEAR)
            st.image([img1, img2])
            st.write("Difference:" + str(final_result[i]))
	#st.image(svd_)
        
        #fr = final_result
        #del fr[ind]
        #print(fr)
        #m = min(fr)
        #ind_m = final_result.index(m)
        #st.write(svd_nms_list[ind_m])
		
	
       

        


    
    

if __name__ == '__main__':
    main()
