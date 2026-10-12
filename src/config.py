import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml


class ExperimentMeta(BaseModel):
    name: str = "rag_configuration_hallucination_benchmark"
    version: str = "1.0.0"
    random_seed: int = 42
    description: str = ""


class DataConfig(BaseModel):
    corpus_path: str = "data/sample/corpus_sample.json"
    queries_path: str = "data/sample/queries_sample.json"
    output_dir: str = "results"
    output_format: str = "json"


class IndependentVariablesConfig(BaseModel):
    top_k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    default_top_k: int = 3
    chunk_sizes: List[int] = Field(default_factory=lambda: [128, 256, 512, 1024])
    chunk_overlap: int = 32
    default_chunk_size: int = 256
    gold_positions: List[str] = Field(default_factory=lambda: ["first", "middle", "last", "auto"])
    default_gold_position: str = "auto"


class RetrieverConfig(BaseModel):
    algorithm: str = "dense_cosine"
    embedding_dim: int = 384
    min_similarity_threshold: float = 0.10


class GeneratorConfig(BaseModel):
    model_name: str = "meta-llama/Llama-3-8B-Instruct"
    mode: str = "deterministic_evaluator"
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 256
    system_prompt: str = (
        "You are a factual QA system. Answer the question using ONLY the provided context."
    )


class ControlledVariablesConfig(BaseModel):
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)


class GuardrailsConfig(BaseModel):
    max_context_tokens: int = 8192
    latency_threshold_ms: float = 2500.0
    max_memory_mb: float = 4096.0
    per_query_timeout_seconds: float = 15.0
    target_error_rate: float = 0.00


class BenchmarkConfig(BaseModel):
    experiment: ExperimentMeta = Field(default_factory=ExperimentMeta)
    data: DataConfig = Field(default_factory=DataConfig)
    independent_variables: IndependentVariablesConfig = Field(
        default_factory=IndependentVariablesConfig
    )
    controlled_variables: ControlledVariablesConfig = Field(
        default_factory=ControlledVariablesConfig
    )
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    metrics: Dict[str, List[str]] = Field(default_factory=dict)


def load_config(config_path: str = "configs/config.yaml") -> BenchmarkConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")

    return BenchmarkConfig(**raw_dict)
