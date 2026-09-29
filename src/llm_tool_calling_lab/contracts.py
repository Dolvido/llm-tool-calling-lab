"""Shared, versioned contracts used by tools, chat, and evaluation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Family = Literal["regression", "classification", "anomaly", "clustering"]

class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = ""
    family: Family
    dataset_id: str
    features: list[str] = Field(min_length=2, max_length=30)
    target: str | None = None
    scoring_dataset_id: str | None = None
    n_clusters: int = Field(default=3, ge=2, le=10)

class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    method_id: str
    task: TaskSpec
    status: Literal["ok", "error"]
    metrics: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0

class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog: Literal["default", "tutorial"] = "default"
    model: str = "qwen3:14b-q4_K_M"
    host: str = "http://localhost:11434"
    max_fits: int = Field(default=2, ge=1, le=2)
    max_tool_calls: int = Field(default=8, ge=1, le=8)
    max_llm_responses: int = Field(default=6, ge=1, le=6)
    max_repairs: int = Field(default=1, ge=0, le=1)
    max_seconds: float = Field(default=180, gt=0, le=180)
    num_ctx: int = Field(default=8192, ge=2048, le=8192)
    max_output_tokens: int = Field(default=1024, ge=1, le=1024)
    max_episode_tokens: int = Field(default=4096, ge=1, le=4096)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 0
