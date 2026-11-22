import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

featurecols=['N','P','K','temperature','humidity','ph','rainfall']

def trainmodel():
    df=pd.read_csv('data/Crop_recommendation.csv')
    x=df[featurecols]
    y=df['label']

    # Turn the crop names into numbers
    encoder=LabelEncoder()
    yenc=encoder.fit_transform(y)

    xtrain,xtest,ytrain,ytest=train_test_split(
        x,yenc,test_size=0.2,random_state=42,stratify=yenc)

    # Standardize using the training mean and std
    means=xtrain.mean()
    stds=xtrain.std()
    xtrainscaled=((xtrain-means)/stds).to_numpy()

    # Random Forest was the best model in our comparison
    model=RandomForestClassifier(n_estimators=200,random_state=42)
    model.fit(xtrainscaled,ytrain)
    return model,means.to_numpy(),stds.to_numpy(),encoder.classes_

model,means,stds,crops=trainmodel()

st.title('AgriSense - Crop Recommendation')
st.write('Enter your soil and climate readings to get the best crop to grow.')

col1,col2=st.columns(2)
with col1:
    nval=st.number_input('Nitrogen (N)',0,140,90)
    pval=st.number_input('Phosphorus (P)',5,145,42)
    kval=st.number_input('Potassium (K)',5,205,43)
    phval=st.number_input('Soil pH',3.0,10.0,6.5)
with col2:
    tempval=st.number_input('Temperature (C)',8.0,44.0,21.0)
    humval=st.number_input('Humidity (%)',14.0,100.0,82.0)
    rainval=st.number_input('Rainfall (mm)',20.0,300.0,203.0)

if st.button('Recommend Crop'):
    reading=np.array([nval,pval,kval,tempval,humval,phval,rainval])

    # Standardize the input the same way as the training data
    scaled=((reading-means)/stds).reshape(1,-1)

    pred=model.predict(scaled)[0]
    crop=crops[pred]
    st.success('Recommended Crop: '+crop.upper())

    # Show the top 3 crops by probability
    probs=model.predict_proba(scaled)[0]
    top=np.argsort(probs)[::-1][:3]
    st.subheader('Top 3 Suggestions')
    for i in top:
        st.write(crops[i].capitalize()+' - '+str(round(probs[i]*100,2))+'%')
