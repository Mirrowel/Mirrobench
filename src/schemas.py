"""
Data models and schemas for the LLM Benchmark system.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ToolDefinition(BaseModel):
    """Definition of a tool for tool-calling benchmarks."""
    name: str
    description: str
    parameters: Dict[str, Any]


class Question(BaseModel):
    """A benchmark question/prompt."""
    id: str
    category: str  # coding, reasoning, tool_calling, writing, etc.
    subcategory: Optional[str] = None
    prompt: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    evaluation_type: Literal["llm_judge", "tool_calling", "code_execution", "exact_match", "contains"]
    evaluation_criteria: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None  # For tool-calling questions
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ModelResponse(BaseModel):
    """Response from a model to a question."""
    question_id: str
    model_name: str
    response_text: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_response: Dict[str, Any]  # Complete API response
    metrics: Dict[str, Any] = Field(default_factory=dict)  # TTFT, TPS, latency, tokens, etc.
    timestamp: str
    error: Optional[str] = None
    # Instance management fields
    instance_id: str  # ISO timestamp identifying this instance
    instance_type: Literal["original", "regenerated", "fixed"] = "original"
    replaces: Optional[str] = None  # instance_id of replaced instance (if any)


class Evaluation(BaseModel):
    """Evaluation result for a model response."""
    question_id: str
    model_name: str
    score: float  # 0-100
    passed: bool
    evaluation_type: str
    evaluator_model: Optional[str] = None  # For LLM-as-judge
    reasoning: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    instance_id: str  # Links to specific response instance


class BenchmarkRun(BaseModel):
    """Metadata for a benchmark run (per model)."""
    run_id: str
    timestamp: str
    model: str  # Single model per run
    categories: List[str]
    total_questions: int
    judge_model: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    run_label: Optional[str] = None  # Custom label for this run


class LeaderboardEntry(BaseModel):
    """Aggregated scores for a model."""
    model_name: str
    overall_score: float
    category_scores: Dict[str, float]
    total_questions: int
    passed_questions: int
    # Performance metrics
    avg_ttft: Optional[float] = None
    avg_tps: Optional[float] = None
    avg_latency: Optional[float] = None
    avg_prompt_tokens: Optional[float] = None
    avg_completion_tokens: Optional[float] = None
    total_tokens_used: Optional[int] = None
    # Cost metrics
    total_cost: Optional[float] = None
    avg_cost_per_question: Optional[float] = None
    cost_efficiency_score: Optional[float] = None


class ExpandedLeaderboardEntry(LeaderboardEntry):
    """Leaderboard entry for 'show all runs' mode with run identification."""
    run_id: str
    run_date: str  # YYYYMMDD format
    run_label: Optional[str] = None
    original_model_name: str  # Model name without run suffix
    is_preferred: bool = False
