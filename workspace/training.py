import argparse, os
import boto3
import json
import pandas as pd
import sys
import numpy as np
import logging



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
import joblib


if __name__ == "__main__":

    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--kernel", type=str, default="linear")
    
    # sm_model_dir: model artifacts stored here after training
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    args, _ = parser.parse_known_args()


    sm_model_dir = args.sm_model_dir
    training_dir = args.train
    testing_dir = args.test

    #define the log level and format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Define a logger
    logger = logging.getLogger(__name__)

    # Read in data
    df_train = pd.read_csv(training_dir + "/dataset_train.csv", sep=",", index_col = "UDI")
    df_test = pd.read_csv(testing_dir + "/dataset_test.csv", sep= ",", index_col = "UDI")
    # Preprocess data
    X_train = df_train.drop(["Target"], axis=1)
    y_train = df_train["Target"]
    X_test = df_test.drop(["Target"], axis=1)
    y_test = df_test["Target"]

    # Build model
    estimator = SVC(C = args.C, kernel = args.kernel)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("accuracy on test is : {}".format(accuracy))
    logger.info("Accuracy : {}".format(accuracy))
    # Save model
    logger.info("saving the model...")
    joblib.dump(estimator, os.path.join(args.sm_model_dir, "model.joblib"))
    logger.info('Training complete.')
    
    
    
    sys.exit(0)