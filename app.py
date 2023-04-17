import streamlit as st
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import getData
import cv2
import numpy as np


labels = ['Atelectasis',
          'Cardiomegaly', 
          'Consolidation',
          'Edema',
          'Effusion',
          'Emphysema',
          'Fibrosis',
          'Hernia',
          'Infiltration',
          'Mass',
          'Nodule',
          'Normal',
          'Pleural_Thickening',
          'Pneumonia',
          'Pneumothorax']

ref = {'Atelectasis': 0.027347086,
       'Cardiomegaly': 0.010086267,
       'Consolidation': 0.0018699166,
       'Edema': 0.0007388931,
       'Effusion': 0.02946913,
       'Emphysema': 0.004008924,
       'Fibrosis': 0.0056709056,
       'Hernia': 0.00016801473,
       'Infiltration': 0.13746345,
       'Mass': 0.00069181505,
       'Nodule': 0.0048904316,
       'Normal': 0.38120866,
       'Pleural_Thickening': 0.0011160322,
       'Pneumonia': 0.00044845155,
       'Pneumothorax': 0.0021950703}



@st.cache_resource
def model():
    base_mobilenet_model = MobileNet(input_shape =  (128, 128, 3), include_top = False, weights = None)
    multi_disease_model = Sequential()
    multi_disease_model.add(base_mobilenet_model)
    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(512))
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(len(labels), activation = 'sigmoid'))
    multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy', 'mae'])

    try:
        multi_disease_model.load_weights('multiClassMobileNet.hdf5')
    except:
        file_id = '11c0YiKInXlTQHy8jfKYhUkO9_bhcLs0C'
        destination = 'multiClassMobileNet.hdf5'
        getData.download_file_from_google_drive(file_id, destination)
        multi_disease_model.load_weights('multiClassMobileNet.hdf5')
    return multi_disease_model


multi_disease_model = model()

st.title('Multiple Disease detection Application')

betaColumnArray = []
imageArray = []

uploadedFiles = st.file_uploader('Upload Chest X-Rays', accept_multiple_files=True)
if uploadedFiles:
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    for upload_file in uploadedFiles:

        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        col1, col2 = st.columns([20, 20])
        col1.image(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC), channels="BGR")
        col1.write(upload_file.name)
        imageArray.append(im)
        betaColumnArray.append(col2)
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if len(imageArray)>0:
    if st.button('Predict'):

        for im, col2 in zip(imageArray, betaColumnArray):

            im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_CUBIC)
            im = im.astype(np.float32)/255.
            im = im.reshape(1, 128, 128, 3)

            res = multi_disease_model.predict(im)
            
            normal = True
            for label, re in zip(labels, res[0]):
                if label != 'NORMAL':
                    if ref[label] >= re:
                        col2.warning(label + ' is positive with : ' +  str(re*100) + ' percent ')
                        normal = False
                    else:
                        col2.success(label + ' is negative.')
            
            if normal:
                col2.write("It looks everything is fine")

       
imageArray.clear()
betaColumnArray.clear()