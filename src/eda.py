import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import featurecols

def runeda(df):
    # Save summary statistics as a csv
    df.describe().to_csv('results/eda/summary_stats.csv')

    # Count of each crop in the dataset
    counts=df['label'].value_counts()
    plt.figure(figsize=(11,5))
    plt.bar(counts.index,counts.values,color='seagreen')
    plt.xticks(rotation=70)
    plt.title('Crop Class Distribution')
    plt.tight_layout()
    plt.savefig('results/eda/class_distribution.png',dpi=120)
    plt.close()

    # Histogram for each input feature
    plt.figure(figsize=(13,8))
    for i,col in enumerate(featurecols):
        plt.subplot(3,3,i+1)
        plt.hist(df[col],bins=20,color='teal')
        plt.title(col)
    plt.tight_layout()
    plt.savefig('results/eda/feature_distributions.png',dpi=120)
    plt.close()

    # Correlation heatmap of the features
    plt.figure(figsize=(8,6))
    sns.heatmap(df[featurecols].corr(),annot=True,cmap='YlGnBu',fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/eda/correlation_heatmap.png',dpi=120)
    plt.close()

    # Boxplot of N, P, K and pH for each crop
    plt.figure(figsize=(14,10))
    for i,col in enumerate(['N','P','K','ph']):
        plt.subplot(2,2,i+1)
        sns.boxplot(data=df,x='label',y=col,color='skyblue')
        plt.xticks(rotation=70)
        plt.title(col+' by Crop')
    plt.tight_layout()
    plt.savefig('results/eda/boxplots_per_crop.png',dpi=110)
    plt.close()

    # Temperature vs humidity scatter
    plt.figure(figsize=(9,6))
    plt.scatter(df['temperature'],df['humidity'],s=12,color='darkorange')
    plt.xlabel('temperature')
    plt.ylabel('humidity')
    plt.title('Temperature vs Humidity')
    plt.tight_layout()
    plt.savefig('results/eda/temp_vs_humidity.png',dpi=120)
    plt.close()

    # Average rainfall needed by each crop
    rain=df.groupby('label')['rainfall'].mean().sort_values()
    plt.figure(figsize=(11,5))
    plt.bar(rain.index,rain.values,color='cornflowerblue')
    plt.xticks(rotation=70)
    plt.ylabel('mean rainfall (mm)')
    plt.title('Average Rainfall per Crop')
    plt.tight_layout()
    plt.savefig('results/eda/rainfall_per_crop.png',dpi=120)
    plt.close()
