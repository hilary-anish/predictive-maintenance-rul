"""
Pydantic schemas for the prediction API.

Why Pydantic schemas instead of raw dicts?
- Auto-validates input (wrong shape/type → clear error, not a crash)
- Auto-generates OpenAPI docs (recruiters see exactly what to send)
- Type hints everywhere (IDE autocomplete, fewer bugs)
- Serialization handled automatically (numpy arrays → JSON)

These schemas are the CONTRACT between your API and its consumers.
"""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Input: sensor readings for one engine unit.

    The client sends a 2D array of shape (seq_len, n_features).
    seq_len = 30 timesteps, n_features = 14 sensors.

    Example:
        {
            "sequence": [[0.5, 0.3, ...], [0.6, 0.4, ...], ...]  # 30 x 14
        }
    """
    sequence: list[list[float]] = Field(
        ...,
        description="Sensor readings: list of 30 timesteps, each with 14 sensor values",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [[0.5] * 14] * 30  # placeholder example
            }
        }


class PredictResponse(BaseModel):
    """Single-unit prediction with conformal intervals."""
    rul: float = Field(..., description="Predicted Remaining Useful Life (cycles)")
    lower: float = Field(..., description="Lower bound of 90% prediction interval")
    upper: float = Field(..., description="Upper bound of 90% prediction interval")
    unit_status: str = Field(..., description="critical / warning / healthy")


class FleetKPIs(BaseModel):
    """Fleet-level summary statistics."""
    total_units: int
    critical: int = Field(..., description="Units with RUL < 30")
    warning: int = Field(..., description="Units with RUL 30-80")
    healthy: int = Field(..., description="Units with RUL > 80")
    avg_rul: float
    min_rul: float


class FleetResponse(BaseModel):
    """Fleet-level predictions for all test units."""
    units: list[int]
    rul_predictions: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    kpis: FleetKPIs


class Alert(BaseModel):
    """Single alert for a unit approaching failure."""
    unit_id: int
    predicted_rul: float
    confidence_lower: float
    confidence_upper: float
    severity: str = Field(..., description="critical (RUL < 15) or warning (RUL < 30)")
    action: str = Field(..., description="Recommended maintenance action")


class AlertResponse(BaseModel):
    """All units below the RUL threshold."""
    threshold: int
    total_alerts: int
    alerts: list[Alert]


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_name: str
    model_version: str
    model_stage: str
