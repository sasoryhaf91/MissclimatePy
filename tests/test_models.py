import pytest

from sklearn.ensemble import RandomForestRegressor

from missclimatepy.models import SUPPORTED_MODELS, make_model

# Detect whether xgboost is available in this environment
try:  # pragma: no cover - environment dependent
    import xgboost  # type: ignore  # noqa: F401

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def test_supported_models_contains_expected_keys():
    """Core model kinds should be present in the registry."""
    required = {"rf", "etr", "gbrt", "hgbt", "linreg", "ridge", "lasso", "knn", "svr", "mlp", "xgb"}
    for key in required:
        assert key in SUPPORTED_MODELS, f"{key} should be in SUPPORTED_MODELS"


def test_make_model_default_is_rf():
    """Calling make_model() without arguments returns a RandomForestRegressor."""
    model = make_model()
    assert isinstance(model, RandomForestRegressor)
    # Check some defaults are there
    assert model.n_estimators == 100
    assert model.random_state == 42


def test_make_model_overrides_default_params():
    """User-provided params must override the defaults."""
    model = make_model("rf", {"n_estimators": 5, "max_depth": 3})
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 5
    assert model.max_depth == 3


def test_make_model_unsupported_kind_raises():
    """Unsupported model kinds should raise a clear ValueError."""
    with pytest.raises(ValueError):
        _ = make_model("not-a-model")


def test_make_model_creates_all_non_xgb_models():
    """
    For every supported kind except 'xgb', make_model should return an
    object with fit/predict methods.
    """
    for kind in SUPPORTED_MODELS.keys():
        if kind == "xgb":
            continue  # handled in a separate test
        model = make_model(kind)
        assert hasattr(model, "fit")
        assert callable(model.fit)
        assert hasattr(model, "predict")
        assert callable(model.predict)


def test_make_model_xgb_behavior_depends_on_environment():
    """
    If xgboost is installed, make_model('xgb') should return a model with fit/predict.
    Otherwise, it should raise ImportError.
    """
    if not HAS_XGB:
        with pytest.raises(ImportError):
            _ = make_model("xgb")
    else:
        model = make_model("xgb")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
