import os

from dotenv import load_dotenv, find_dotenv
from sklearn.externals import joblib

from dsflow.pipeline import make_ds_flow_pipeline
from dsflow.preprocessing import (create_dataset,
                                  get_labeled_comments,
                                  get_token_weigths)

COMMENTS_URL = 'https://jsonplaceholder.typicode.com/comments'
WEIGHTS_CSV = 'dsflow/token_weights.csv'
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    comments = get_labeled_comments(
        COMMENTS_URL,
        os.getenv('DB_USERNAME'),
        os.getenv('DB_USERNAME'),
        os.getenv('DB_TOKEN')
    )
    weights = get_token_weigths(WEIGHTS_CSV)
    comments = create_dataset(comments, weights)
    model_pipeline = make_ds_flow_pipeline()
    model_pipeline = model_pipeline.fit(
        comments['body'],
        comments['sentiment'],
        forest__sample_weight=comments['weights']
    )
    joblib.dump(model_pipeline, os.path.join(SAVE_DIR, 'model.pkl'))


if __name__ == "__main__":
    load_dotenv(find_dotenv(), verbose=True)
    main()
