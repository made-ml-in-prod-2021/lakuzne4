from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             average_precision_score,
                             roc_auc_score,
                             )
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

possible_models_dict = {
    'LogisticRegression': LogisticRegression,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'RandomForestClassifier': RandomForestClassifier,
}

possible_score_funcs = {
    'accuracy_score': accuracy_score,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'roc_auc_score': roc_auc_score,
    'average_precision_score': average_precision_score,
}
