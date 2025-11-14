"""
Results manager for saving and retrieving benchmark results.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from src.schemas import ModelResponse, Evaluation, BenchmarkRun, LeaderboardEntry


class ResultsManager:
    """Manages benchmark results storage and retrieval."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[BenchmarkRun] = None
        self.current_run_dir: Optional[Path] = None
        self.leaderboard_prefs_path = self.results_dir / "leaderboard_preferences.json"

    def create_run(
        self,
        model: str,
        categories: List[str],
        total_questions: int,
        judge_model: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> str:
        """Create a new benchmark run for a single model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include model name in run_id for clarity
        model_slug = self._sanitize_name(model)
        run_id = f"{model_slug}_{timestamp}"

        self.current_run = BenchmarkRun(
            run_id=run_id,
            timestamp=timestamp,
            model=model,
            categories=categories,
            total_questions=total_questions,
            judge_model=judge_model,
            config=config or {}
        )

        # Create run directory structure
        self.current_run_dir = self.results_dir / run_id
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        (self.current_run_dir / "responses").mkdir(exist_ok=True)
        (self.current_run_dir / "evaluations").mkdir(exist_ok=True)

        # Save metadata
        self._save_metadata()

        return run_id

    def _save_metadata(self):
        """Save run metadata to file."""
        if self.current_run and self.current_run_dir:
            metadata_path = self.current_run_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_run.model_dump(), f, indent=2)

    def save_response(self, response: ModelResponse, version: Optional[str] = None):
        """
        Save a model response with versioning support.

        Args:
            response: The ModelResponse to save
            version: Optional version identifier. If None, uses timestamp
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        # Create model directory if it doesn't exist
        model_dir = self.current_run_dir / "responses" / self._sanitize_name(response.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create question directory for versioned responses
        question_dir = model_dir / response.question_id
        question_dir.mkdir(exist_ok=True)

        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save versioned response
        response_path = question_dir / f"v{version}.json"
        response_data = response.model_dump()
        response_data['version'] = version
        response_data['created_at'] = datetime.now().isoformat()

        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)

        # Update latest.json pointer
        latest_path = question_dir / "latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump({'version': version, 'file': f"v{version}.json"}, f, indent=2)

    def save_evaluation(self, evaluation: Evaluation):
        """Save an evaluation result. Supports multiple evaluation types per question."""
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        # Create model directory if it doesn't exist
        model_dir = self.current_run_dir / "evaluations" / self._sanitize_name(evaluation.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation with type suffix to support multiple evaluations
        eval_filename = f"{evaluation.question_id}_{evaluation.evaluation_type}.json"
        eval_path = model_dir / eval_filename
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation.model_dump(), f, indent=2, ensure_ascii=False)

    def calculate_and_save_scores(self, questions: List):
        """Calculate aggregate scores and save to scores.json."""
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        evaluations_dir = self.current_run_dir / "evaluations"
        responses_dir = self.current_run_dir / "responses"

        leaderboard: Dict[str, Dict] = {}

        # Collect all evaluations
        for model_dir in evaluations_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name

                evaluations = []
                for eval_file in model_dir.glob("*.json"):
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        evaluations.append(Evaluation(**eval_data))

                # Collect metrics from responses
                metrics = []
                model_response_dir = responses_dir / model_name
                if model_response_dir.exists():
                    # Handle both versioned (subdirectories) and legacy (flat) structures
                    question_ids = set()

                    # Check for versioned structure
                    for question_dir in model_response_dir.iterdir():
                        if question_dir.is_dir():
                            question_ids.add(question_dir.name)

                    # Check for legacy flat structure
                    for response_file in model_response_dir.glob("*.json"):
                        if '_fixed' not in response_file.stem:
                            question_ids.add(response_file.stem)

                    # Load metrics from each response
                    for question_id in question_ids:
                        # Get response (uses get_response which handles both structures)
                        try:
                            response_path = None

                            # Try versioned structure first
                            question_dir = model_response_dir / question_id
                            if question_dir.exists() and question_dir.is_dir():
                                latest_path = question_dir / "latest.json"
                                if latest_path.exists():
                                    with open(latest_path, 'r', encoding='utf-8') as f:
                                        latest_info = json.load(f)
                                        response_path = question_dir / latest_info['file']
                                else:
                                    # Find most recent version
                                    version_files = sorted(question_dir.glob("v*.json"), reverse=True)
                                    if version_files:
                                        response_path = version_files[0]

                            # Fall back to legacy flat structure
                            if not response_path:
                                legacy_path = model_response_dir / f"{question_id}.json"
                                if legacy_path.exists():
                                    response_path = legacy_path

                            if response_path:
                                with open(response_path, 'r', encoding='utf-8') as f:
                                    response_data = json.load(f)
                                    if response_data.get('metrics'):
                                        metrics.append(response_data['metrics'])
                        except Exception as e:
                            # Skip responses that can't be loaded
                            continue

                # Calculate scores
                if evaluations:
                    overall_score = sum(e.score for e in evaluations) / len(evaluations)
                    passed = sum(1 for e in evaluations if e.passed)

                    # Category breakdown
                    category_scores: Dict[str, List[float]] = defaultdict(list)
                    for evaluation in evaluations:
                        # Find question category
                        question = next((q for q in questions if q.id == evaluation.question_id), None)
                        if question:
                            category_scores[question.category].append(evaluation.score)

                    category_averages = {
                        cat: sum(scores) / len(scores)
                        for cat, scores in category_scores.items()
                    }

                    # Calculate average metrics
                    avg_ttft = None
                    avg_tps = None
                    avg_latency = None

                    if metrics:
                        ttfts = [m.get('ttft') for m in metrics if m.get('ttft') is not None]
                        tpss = [m.get('tokens_per_second') for m in metrics if m.get('tokens_per_second') is not None]
                        latencies = [m.get('total_latency') for m in metrics if m.get('total_latency') is not None]

                        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
                        avg_tps = sum(tpss) / len(tpss) if tpss else None
                        avg_latency = sum(latencies) / len(latencies) if latencies else None

                    entry = LeaderboardEntry(
                        model_name=model_name,
                        overall_score=overall_score,
                        category_scores=category_averages,
                        total_questions=len(evaluations),
                        passed_questions=passed,
                        avg_ttft=avg_ttft,
                        avg_tps=avg_tps,
                        avg_latency=avg_latency
                    )

                    leaderboard[model_name] = entry.model_dump()

        # Sort by overall score
        sorted_leaderboard = dict(
            sorted(leaderboard.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        )

        # Save scores
        scores_path = self.current_run_dir / "scores.json"
        with open(scores_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_leaderboard, f, indent=2)

        return sorted_leaderboard

    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Load a specific benchmark run."""
        run_dir = self.results_dir / run_id
        if not run_dir.exists():
            return None

        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return BenchmarkRun(**data)

    def get_all_runs(self) -> List[BenchmarkRun]:
        """Get all benchmark runs."""
        runs = []
        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        runs.append(BenchmarkRun(**data))

        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x.timestamp, reverse=True)
        return runs

    def get_response(self, run_id: str, model_name: str, question_id: str, use_fixed: bool = False, version: Optional[str] = None) -> Optional[ModelResponse]:
        """
        Get a specific model response with version support.

        Args:
            run_id: The benchmark run ID
            model_name: The model name
            question_id: The question ID
            use_fixed: If True, try to load fixed version first (legacy)
            version: Specific version to load. If None, loads latest.

        Returns:
            ModelResponse or None
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        # Check if question directory exists (new versioned structure)
        if question_dir.exists() and question_dir.is_dir():
            if version:
                # Load specific version
                version_path = question_dir / f"v{version}.json"
                if not version_path.exists():
                    return None
            else:
                # Load latest version
                latest_path = question_dir / "latest.json"
                if latest_path.exists():
                    with open(latest_path, 'r', encoding='utf-8') as f:
                        latest_info = json.load(f)
                        version_path = question_dir / latest_info['file']
                else:
                    # If no latest.json, find the most recent version
                    version_files = sorted(question_dir.glob("v*.json"), reverse=True)
                    if not version_files:
                        return None
                    version_path = version_files[0]

            with open(version_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ModelResponse(**data)

        # Legacy: Try fixed version first if requested
        if use_fixed:
            fixed_path = base_path / f"{question_id}_fixed.json"
            if fixed_path.exists():
                with open(fixed_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ModelResponse(**data)

        # Legacy: Fall back to original flat structure
        response_path = base_path / f"{question_id}.json"
        if not response_path.exists():
            return None

        with open(response_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return ModelResponse(**data)

    def has_fixed_response(self, run_id: str, model_name: str, question_id: str) -> bool:
        """Check if a fixed response exists."""
        fixed_path = (
            self.results_dir / run_id / "responses" /
            self._sanitize_name(model_name) / f"{question_id}_fixed.json"
        )
        return fixed_path.exists()

    def save_fixed_response(self, response: ModelResponse):
        """Save a fixed version of a response."""
        if not self.current_run_dir:
            # For manual fixes, need to find the run directory
            # This will be called from API with explicit run_id
            raise RuntimeError("No active run. Use save_fixed_response_for_run() instead.")

        # Create model directory if it doesn't exist
        model_dir = self.current_run_dir / "responses" / self._sanitize_name(response.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save fixed response
        response_path = model_dir / f"{response.question_id}_fixed.json"
        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)

    def save_fixed_response_for_run(self, run_id: str, response: ModelResponse):
        """Save a fixed response for a specific run (used by API)."""
        run_dir = self.results_dir / run_id

        # Create model directory if it doesn't exist
        model_dir = run_dir / "responses" / self._sanitize_name(response.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save fixed response
        response_path = model_dir / f"{response.question_id}_fixed.json"
        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)

    def get_evaluation(self, run_id: str, model_name: str, question_id: str, evaluation_type: str = None) -> Optional[Evaluation]:
        """
        Get a specific evaluation.
        If evaluation_type is provided, gets that specific type.
        Otherwise, returns llm_judge evaluation if available, or the first evaluation found.
        """
        eval_dir = self.results_dir / run_id / "evaluations" / self._sanitize_name(model_name)

        if not eval_dir.exists():
            return None

        if evaluation_type:
            # Get specific type
            eval_path = eval_dir / f"{question_id}_{evaluation_type}.json"
            if eval_path.exists():
                with open(eval_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return Evaluation(**data)
            return None

        # Prefer llm_judge evaluation for backward compatibility
        judge_path = eval_dir / f"{question_id}_llm_judge.json"
        if judge_path.exists():
            with open(judge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Evaluation(**data)

        # Fall back to old naming format (no type suffix)
        old_path = eval_dir / f"{question_id}.json"
        if old_path.exists():
            with open(old_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Evaluation(**data)

        # Find any evaluation for this question
        for eval_file in eval_dir.glob(f"{question_id}_*.json"):
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Evaluation(**data)

        return None

    def get_all_evaluations(self, run_id: str, model_name: str, question_id: str) -> List[Evaluation]:
        """Get all evaluations for a specific question (all types)."""
        eval_dir = self.results_dir / run_id / "evaluations" / self._sanitize_name(model_name)

        if not eval_dir.exists():
            return []

        evaluations = []

        # Check for old format (no type suffix)
        old_path = eval_dir / f"{question_id}.json"
        if old_path.exists():
            with open(old_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                evaluations.append(Evaluation(**data))

        # Get all evaluations with type suffix
        for eval_file in eval_dir.glob(f"{question_id}_*.json"):
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                evaluations.append(Evaluation(**data))

        return evaluations

    def get_leaderboard(self, run_id: str) -> Optional[Dict]:
        """Get the leaderboard for a run."""
        scores_path = self.results_dir / run_id / "scores.json"

        if not scores_path.exists():
            return None

        with open(scores_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_model_runs(self, model_name: str) -> List[Dict]:
        """
        Get all benchmark runs that contain a specific model.

        Returns:
            List of dicts with run info and model scores
        """
        model_runs = []
        sanitized_name = self._sanitize_name(model_name)

        for run_dir in self.results_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if this run has responses for the model
            model_response_dir = run_dir / "responses" / sanitized_name
            if not model_response_dir.exists():
                continue

            # Load run metadata
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                run_data = json.load(f)

            # Load model scores from leaderboard
            scores_path = run_dir / "scores.json"
            model_score = None
            if scores_path.exists():
                with open(scores_path, 'r', encoding='utf-8') as f:
                    leaderboard = json.load(f)
                    model_score = leaderboard.get(model_name)

            model_runs.append({
                "run_id": run_data["run_id"],
                "timestamp": run_data["timestamp"],
                "categories": run_data.get("categories", []),
                "total_questions": run_data.get("total_questions", 0),
                "score_data": model_score
            })

        # Sort by timestamp (newest first)
        model_runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return model_runs

    def get_leaderboard_preferences(self) -> Dict[str, str]:
        """
        Get leaderboard preferences (which run_id to use per model).

        Returns:
            Dict mapping model_name to run_id
        """
        if not self.leaderboard_prefs_path.exists():
            return {}

        with open(self.leaderboard_prefs_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def set_leaderboard_preference(self, model_name: str, run_id: str):
        """
        Set which run to use for a model in the leaderboard.

        Args:
            model_name: The model name
            run_id: The run_id to use for this model in leaderboard
        """
        preferences = self.get_leaderboard_preferences()
        preferences[model_name] = run_id

        with open(self.leaderboard_prefs_path, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2)

    def clear_leaderboard_preference(self, model_name: str):
        """
        Clear leaderboard preference for a model (will use latest run).

        Args:
            model_name: The model name
        """
        preferences = self.get_leaderboard_preferences()
        if model_name in preferences:
            del preferences[model_name]
            with open(self.leaderboard_prefs_path, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, indent=2)

    def get_unified_leaderboard(self) -> Dict[str, Dict]:
        """
        Get a leaderboard using preferred runs for each model.
        Uses latest run for models without a preference.

        Returns:
            Dict mapping model_name to leaderboard entry with run_id included
        """
        preferences = self.get_leaderboard_preferences()
        leaderboard = {}

        # Get all unique models across all runs
        all_models = set()
        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "responses").exists():
                for model_dir in (run_dir / "responses").iterdir():
                    if model_dir.is_dir():
                        # Reverse sanitization to get original model name
                        # This is a simple approach - we'll get the actual name from metadata
                        pass

        # Collect model names from all runs
        all_runs = self.get_all_runs()
        for run in all_runs:
            all_models.add(run.model)

        # For each model, get its leaderboard entry from preferred or latest run
        for model_name in all_models:
            # Check if there's a preference
            if model_name in preferences:
                run_id = preferences[model_name]
            else:
                # Use latest run containing this model
                model_runs = self.get_model_runs(model_name)
                if not model_runs:
                    continue
                run_id = model_runs[0]["run_id"]  # Already sorted by timestamp, newest first

            # Get leaderboard data for this model from the specified run
            run_leaderboard = self.get_leaderboard(run_id)
            if run_leaderboard:
                # scores.json uses sanitized model names as keys (slashes -> underscores)
                sanitized_model_name = self._sanitize_name(model_name)

                # Try both original and sanitized names for backwards compatibility
                if model_name in run_leaderboard:
                    entry = run_leaderboard[model_name].copy()
                elif sanitized_model_name in run_leaderboard:
                    entry = run_leaderboard[sanitized_model_name].copy()
                else:
                    continue

                entry["run_id"] = run_id
                entry["is_preferred"] = model_name in preferences
                leaderboard[model_name] = entry

        # Sort by overall score
        sorted_leaderboard = dict(
            sorted(leaderboard.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        )

        return sorted_leaderboard

    def list_response_versions(self, run_id: str, model_name: str, question_id: str) -> List[Dict]:
        """
        List all versions of a response for a specific question.

        Returns list of version info dicts with keys: version, created_at, is_latest
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        if not question_dir.exists() or not question_dir.is_dir():
            # Legacy structure - check if single file exists
            response_path = base_path / f"{question_id}.json"
            if response_path.exists():
                # Return single version
                return [{
                    'version': 'legacy',
                    'created_at': datetime.fromtimestamp(response_path.stat().st_mtime).isoformat(),
                    'is_latest': True
                }]
            return []

        # Get latest version
        latest_version = None
        latest_path = question_dir / "latest.json"
        if latest_path.exists():
            with open(latest_path, 'r', encoding='utf-8') as f:
                latest_info = json.load(f)
                latest_version = latest_info.get('version')

        # Find all version files
        versions = []
        for version_file in sorted(question_dir.glob("v*.json"), reverse=True):
            if version_file.name == 'latest.json':
                continue

            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                version_id = data.get('version', version_file.stem[1:])  # Remove 'v' prefix
                created_at = data.get('created_at', datetime.fromtimestamp(version_file.stat().st_mtime).isoformat())

                versions.append({
                    'version': version_id,
                    'created_at': created_at,
                    'is_latest': version_id == latest_version,
                    'file': version_file.name
                })

        return versions

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize model name for use in file paths."""
        return name.replace('/', '_').replace('\\', '_').replace(':', '_')
