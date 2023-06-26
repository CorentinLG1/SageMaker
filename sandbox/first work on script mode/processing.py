import argparse
import os
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = "/opt/ml/processing/input/dataset.csv"

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path, index_col = "UDI")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    mapping = {'L' : 0, 'M' : 1, 'H' : 2}
    df["Type"] = df["Type"].map(mapping)
    df = df.iloc[:,1:-1]
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    df_train, df_test = train_test_split(df,test_size=split_ratio, random_state=0)
    print(df_train)
    
    dataset_train = "train/dataset_train.csv"
    dataset_test = "test/dataset_test.csv"
    output_data_path = "/opt/ml/processing/"
    
    df_train.to_csv(os.path.join(output_data_path,dataset_train))
    df_test.to_csv(os.path.join(output_data_path,dataset_test))
    
    sys.exit(0)