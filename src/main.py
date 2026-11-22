from preprocessing import loaddata,cleandata,preparedata
from eda import runeda
from training import trainmodels
from evaluation import evaluatemodels

def runpipeline():
    print('Loading dataset...')
    df=loaddata()
    print('Shape:',df.shape,'| Crops:',df['label'].nunique())

    df,missing,removed=cleandata(df)
    print('Missing values:',missing,'| Duplicates removed:',removed)

    print('Running exploratory data analysis...')
    runeda(df)

    print('Preparing data (encode, standardize, split)...')
    xtrain,xtest,ytrain,ytest,encoder=preparedata(df)
    print('Train rows:',len(xtrain),'| Test rows:',len(xtest))

    print('Training models...')
    models=trainmodels(xtrain,ytrain)

    print('Evaluating models...')
    table,bestname=evaluatemodels(models,xtest,ytest,encoder)

    print('\nModel comparison:')
    print(table.to_string(index=False))
    print('\nBest model by F1 score:',bestname)
    print('Done. Results and plots are in the results/ folder.')
    return table,bestname

if __name__=='__main__':
    runpipeline()
