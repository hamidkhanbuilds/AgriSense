import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

featurecols=['N','P','K','temperature','humidity','ph','rainfall']

def loaddata():
    df=pd.read_csv('data/Crop_recommendation.csv')
    return df

def cleandata(df):
    # Count missing values and drop duplicate rows
    missing=df.isnull().sum().sum()
    before=len(df)
    df=df.drop_duplicates()
    removed=before-len(df)
    return df,missing,removed

def preparedata(df):
    x=df[featurecols]
    y=df['label']

    # Encode the 22 crop names into numbers
    encoder=LabelEncoder()
    yenc=encoder.fit_transform(y)

    xtrain,xtest,ytrain,ytest=train_test_split(
        x,yenc,test_size=0.2,random_state=42,stratify=yenc)

    # Standardize using the training mean and std
    means=xtrain.mean()
    stds=xtrain.std()
    xtrainscaled=((xtrain-means)/stds).to_numpy()
    xtestscaled=((xtest-means)/stds).to_numpy()

    return xtrainscaled,xtestscaled,ytrain,ytest,encoder
