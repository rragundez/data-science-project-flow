from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def make_ds_flow_pipeline(vocabulary_size=None, seed=None):
    vectorizer = CountVectorizer(max_features=vocabulary_size)
    forest = RandomForestClassifier(random_state=seed)
    pipeline = Pipeline([
        ('count_vectorizer', vectorizer),
        ('forest', forest)
    ])
    return pipeline
