import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_reduced_plot(df):
    '''
    Reduce features to n_components=2 and plot the data
    df: 2D dataframe with last column as targets and other columns are numerical features
    '''
    df_X = df.iloc[:,:-1]
    df_Y = df.iloc[:,-1]

    classes = df_Y.unique()

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(df_X.values)
    X_reduced_df = pd.DataFrame(X_reduced, columns = ['PC1', 'PC2'])
    
    # reset index before concat ref:https://stackoverflow.com/questions/40339886/pandas-concat-generates-nan-values
    X_reduced_df.reset_index(drop=True, inplace=True) 
    df_Y.reset_index(drop=True, inplace=True) 

    concat_df = pd.concat([X_reduced_df, df_Y], axis = 1)

    # plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 

    for classs in classes:
        class_indices = concat_df.iloc[:,-1] == classs
        ax.scatter(concat_df.loc[class_indices, 'PC1'],
                    concat_df.loc[class_indices, 'PC2'],
                    s = 50)
    ax.legend(classes)
    ax.grid()
    plt.show()