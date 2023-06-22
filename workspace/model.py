import pandas as pd
import numpy as np
import os
import argparse

from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(data_path,header=0, index=None, col_to_drop=None) :
    print("loading data...")
    dataset = pd.read_csv(data_path,header=header,index_col=index)
    if col_to_drop is not None :
        dataset=dataset.drop(col_to_drop,axis=1)
    return dataset

def show_class_repartition(target) :
    c1 = len(target)
    c2 = len(np.where(target==1)[0])
    print("The class proportion is :",c2/c1)
    return c2/c1

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--kernel", type=str, default="linear")

    args, _ = parser.parse_known_args()

    #load the data
    dataset = load_data("predictive_maintenance.csv",col_to_drop=["UDI","Product ID","Failure Type"])

    #encode the colone type
    mapping = {'L' : 0, 'M' : 1, 'H' : 2}
    dataset["Type"] = dataset["Type"].map(mapping)

    #show the balance between the two class
    output = dataset["Target"]
    output = output.to_numpy()
    show_class_repartition(output)

    #split the data in testing and training
    X = dataset.drop(["Target"],axis=1)
    y = dataset["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state=42)
    #resample the dataset
    smote = SMOTE(random_state=0)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)
    show_class_repartition(y_train_resampled)


    scaler = StandardScaler()
    scaler.fit_transform(X_train_resampled)
    scaler.transform(X_test)
    
    #train the SVM on the dataset
    print(f"training an SVC using C={args.C} and a {args.kernel} kernel")
    model = SVC(C = args.C, kernel=args.kernel)

    model.fit(X_train_resampled,y_train_resampled)
    y_pred = model.predict(X_test)
    print("accuracy : ", accuracy_score(y_test,y_pred))
    print()

    #example of prediction
    index = np.where(y_test==1)
    i = 8
    print("input : ", X_test.iloc[index[0][i]].tolist(), "target :", y_test.iloc[index[0][i]])
    print("prediction : ", model.predict([X_test.iloc[index[0][i]]]))

