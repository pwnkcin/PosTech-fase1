import numpy as np

from src.data import build_pipeline, load_data


def test_load_data_coerces_the_known_dirty_amh_value_to_nan():
    X, y = load_data()

    assert X["AMH"].isna().sum() == 1
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}


def test_build_pipeline_imputes_and_standardizes():
    X, _ = load_data()
    pipeline = build_pipeline()

    transformed = pipeline.fit_transform(X)

    assert not np.isnan(transformed).any()
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(transformed.std(axis=0), 1, atol=1e-6)
