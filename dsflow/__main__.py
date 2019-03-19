import os

from dotenv import load_dotenv, find_dotenv
from sklearn.externals import joblib

from dsflow.pipeline import make_ds_flow_pipeline
from dsflow.preprocessing import (create_dataset,
                                  get_labeled_comments,
                                  get_token_weigths)

from dsflow.configuration import CONFIG


def main():
    comments = get_labeled_comments(
        CONFIG['comments_url'],
        os.getenv('DB_USERNAME'),
        os.getenv('DB_USERNAME'),
        os.getenv('DB_TOKEN')
    )
    weights = get_token_weigths(CONFIG['weights_csv'])
    comments = create_dataset(comments, weights)
    model_pipeline = make_ds_flow_pipeline()
    model_pipeline = model_pipeline.fit(
        comments['body'],
        comments['sentiment'],
        forest__sample_weight=comments['weights']
    )
    joblib.dump(model_pipeline, os.path.join(
        CONFIG['model_dir'], 'model.pkl'
    ))


if __name__ == "__main__":
    load_dotenv(find_dotenv(), verbose=True)
    main()
