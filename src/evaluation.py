import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
    f1_score,confusion_matrix,classification_report)

def savematrix(name,ytest,ypred,crops):
    cm=confusion_matrix(ytest,ypred)
    plt.figure(figsize=(11,9))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Greens',
        xticklabels=crops,yticklabels=crops,cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - '+name)
    plt.xticks(rotation=70)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/confusion/'+name.lower().replace(' ','')+'.png',dpi=110)
    plt.close()

def comparisonchart(table):
    metrics=['Accuracy','Precision','Recall','F1 Score']
    plt.figure(figsize=(10,6))
    xpos=range(len(table))
    width=0.2
    for i,m in enumerate(metrics):
        pos=[p+i*width for p in xpos]
        plt.bar(pos,table[m],width,label=m)
    plt.xticks([p+1.5*width for p in xpos],table['Model'],rotation=15)
    plt.ylim(0.85,1.01)
    plt.ylabel('score')
    plt.title('Model Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/model_comparison.png',dpi=120)
    plt.close()

def evaluatemodels(models,xtest,ytest,encoder):
    crops=encoder.classes_
    rows=[]
    reports=[]
    for name,model in models.items():
        ypred=model.predict(xtest)
        acc=accuracy_score(ytest,ypred)
        prec=precision_score(ytest,ypred,average='macro')
        rec=recall_score(ytest,ypred,average='macro')
        f1=f1_score(ytest,ypred,average='macro')
        rows.append([name,acc,prec,rec,f1])
        savematrix(name,ytest,ypred,crops)
        reports.append('===== '+name+' =====')
        reports.append(classification_report(ytest,ypred,target_names=crops))

    table=pd.DataFrame(rows,columns=['Model','Accuracy','Precision','Recall','F1 Score'])
    table=table.sort_values('F1 Score',ascending=False).reset_index(drop=True)
    table.to_csv('results/metrics.csv',index=False)

    with open('results/classification_reports.txt','w') as f:
        f.write('\n'.join(reports))

    comparisonchart(table)

    # Best model by highest F1 score
    bestname=table.iloc[0]['Model']
    return table,bestname
