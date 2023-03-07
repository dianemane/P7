import pytest
from process_utils import *
my_csv_path = "feature_engineered_data_subset.csv"
model_path = "best_model_balanced.pkl"

data = pd.read_csv(my_csv_path, encoding='latin1', index_col=0)
best_model_balanced = pickle.load(open(model_path, 'rb'))


def test_final_model():

    X = df_to_X_preprocessing(data, 305317)
    thr, proba = final_model(X, best_model_balanced, 0.5)
    expected_data_type = (int, float, numpy.float64)

    assert type( proba[0] ) in expected_data_type


def test_apply_threshold():

    X = df_to_X_preprocessing(data, 305317)
    pred = apply_threshold([0.1,0.2,0.6], 0.5)
    expected_data_type = (int, float, numpy.float64)

    assert pred == [0,0,1]
