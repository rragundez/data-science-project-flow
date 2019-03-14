import requests

import numpy as np
import pandas as pd


def get_comments(comments_url, username, password, token):
    comments = requests.get(comments_url).json()
    return pd.DataFrame(comments).drop(columns=['id'])

def get_labeled_comments(comments_url, username, password, token):
    comments = get_comments(comments_url, username, password, token)
    comments['sentiment'] = np.random.randint(0, 2, size=len(comments))
    return comments

def get_token_weigths(path):
    return pd.read_csv(path)

def _add_weight(x, tokens, weights):
    try:
        idx = tokens.index(x)
    except:
        return 1
    else:
        return weights[idx]

def create_dataset(comments, token_weights):
    comments['weights'] = comments['name'].apply(lambda x: _add_weight(x, token_weights['token'].tolist(), token_weights['weight'].tolist()))
    return comments
