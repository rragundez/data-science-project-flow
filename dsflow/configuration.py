import os

CONFIG = {
    'comments_url': 'https://jsonplaceholder.typicode.com/comments',
    'weights_csv': 'dsflow/token_weights.csv',
    'model_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
}
