import argparse, os
import boto3
import json
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals import joblib

if __name__ == "__main__":

    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--kernel", type=str, default="linear")
    
    # sm_model_dir: model artifacts stored here after training
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    args, _ = parser.parse_known_args()

    model_dir = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir = args.train
    testing_dir = args.test
    # Read in data
    df_train = pd.read_csv(training_dir + "/dataset_train.csv", sep=",", index_col = "UDI")
    df_test = pd.read_csv(testing_dir + "/dataset_test.csv", sep= ",", index_col = "UDI")
    # Preprocess data
    X_train = df_train.drop(["Target"], axis=1)
    y_train = df_train["Target"]
    X_test = df_test.drop(["Target"], axis=1)
    y_test = df_test["Target"]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build model
    estimator = SVC(C = args.C, kernel = args.kernel)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print("accuracy on test is : {}".format(accuracy_score(y_test,y_pred)))
    # Save model
    joblib.dump(estimator, os.path.join(args.sm_model_dir, "model.joblib"))
    
    sys.exit(0)
  


    
    
# Model serving
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_body = json.loads(request_body)
        inpVar = request_body["Input"]
        return inpVar
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""
def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {"Output": res}
    return respJSON

