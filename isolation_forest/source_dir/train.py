from __future__ import print_function

import argparse
import os
from io import StringIO

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from custom_tranforms import get_pipeline

def shingle(data, shingle_size):
    num_data = len(data)
    shingled_data = np.zeros((num_data - shingle_size, shingle_size))

    for n in range(num_data - shingle_size):
        shingled_data[n] = data[n : (n + shingle_size)]
    return shingled_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shingle_size", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    
    args = parser.parse_args()
    
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        print(f)
    
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    
    raw_data = [pd.read_csv(file, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)
    
    
    # single data with shingle size=48 (one day)
    if args.shingle_size > 1:
        input_data = shingle(train_data.values[:, 1], args.shingle_size)
    else:
        input_data = train_data.value.to_numpy().reshape(-1, 1)
        
    clf = IsolationForest(max_samples=args.max_samples, random_state=args.random_state)
    clf.fit(input_data)
    
    result_df = pd.DataFrame(input_data)
    # compute metrics
    y_pred_train = clf.predict(input_data)
    y_score_train = clf.score_samples(input_data)
    
    result_df['prediction'] = y_pred_train
    result_df['score'] = y_score_train

    anomalies = result_df[result_df['prediction']==-1]['prediction'].count()
    total = result_df.shape[0]
    
    score_mean = result_df["score"].mean()
    score_std = result_df["score"].std()
    
    print(f"#_anomalies = {anomalies}; pct_anomalies = {anomalies/total*100} %;")
    
    print(f"avg_avgscore = {score_mean:.2f}; std = {score_std: .2f};")
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        
        pipe = get_pipeline(column='rnd_column')

        return pipe.fit_transform(df)
    else:
        raise ValueError("{} not supported by script!".format(content_type))    
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf