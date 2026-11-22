from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def buildmodels():
    models={
        'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42),
        'Decision Tree':DecisionTreeClassifier(random_state=42),
        'KNN':KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes':GaussianNB()
    }
    return models

def trainmodels(xtrain,ytrain):
    models=buildmodels()
    for name,model in models.items():
        model.fit(xtrain,ytrain)
    return models
