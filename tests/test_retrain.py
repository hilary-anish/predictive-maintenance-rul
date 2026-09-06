"""
Test the automated retraining pipeline logic.

Why test the pipeline? The retrain pipeline makes critical decisions:
promote or keep. Wrong logic = deploying bad models or missing
improvements. These tests verify the decision logic in isolation.
"""
from unittest.mock import patch, MagicMock
from pdm.pipeline.retrain import compare_and_promote, RMSE_IMPROVEMENT_MIN


@patch("pdm.pipeline.retrain.MlflowClient")
@patch("pdm.pipeline.retrain.mlflow")
def test_promote_when_no_champion(mock_mlflow, mock_client_cls):
    """First model should always be promoted."""
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(version="1")]
    mock_client_cls.return_value = mock_client
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock()
    ))

    challenger = {"rmse": 13.5, "mae": 10.0, "phm_score": 300.0}
    result = compare_and_promote(None, challenger)
    assert result["decision"] == "promote"


@patch("pdm.pipeline.retrain.MlflowClient")
@patch("pdm.pipeline.retrain.mlflow")
def test_promote_when_challenger_better(mock_mlflow, mock_client_cls):
    """Challenger beating champion by >= threshold should be promoted."""
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(version="2")]
    mock_client_cls.return_value = mock_client
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock()
    ))

    champion = {"rmse": 14.0, "mae": 10.5, "phm_score": 350.0}
    challenger = {"rmse": 13.0, "mae": 9.5, "phm_score": 270.0}
    # Improvement = 14.0 - 13.0 = 1.0 >= 0.5
    result = compare_and_promote(champion, challenger)
    assert result["decision"] == "promote"


@patch("pdm.pipeline.retrain.MlflowClient")
@patch("pdm.pipeline.retrain.mlflow")
def test_keep_champion_when_not_enough_improvement(mock_mlflow, mock_client_cls):
    """Challenger not beating champion by enough should be rejected."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock()
    ))

    champion = {"rmse": 13.5, "mae": 10.0, "phm_score": 300.0}
    challenger = {"rmse": 13.3, "mae": 9.8, "phm_score": 290.0}
    # Improvement = 13.5 - 13.3 = 0.2 < 0.5
    result = compare_and_promote(champion, challenger)
    assert result["decision"] == "keep_champion"


@patch("pdm.pipeline.retrain.MlflowClient")
@patch("pdm.pipeline.retrain.mlflow")
def test_keep_champion_when_challenger_worse(mock_mlflow, mock_client_cls):
    """Challenger worse than champion should definitely be rejected."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock()
    ))

    champion = {"rmse": 13.0, "mae": 9.5, "phm_score": 270.0}
    challenger = {"rmse": 15.0, "mae": 11.0, "phm_score": 370.0}
    result = compare_and_promote(champion, challenger)
    assert result["decision"] == "keep_champion"
