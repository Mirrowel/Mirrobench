"""
Results manager for saving and retrieving benchmark results.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from src.schemas import ModelResponse, Evaluation, BenchmarkRun, LeaderboardEntry

logger = logging.getLogger(__name__)


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

    def save_response(self, response: ModelResponse, set_as_current: bool = True):
        """
        Save a model response with instance-based storage.

        Args:
            response: The ModelResponse to save (must have instance_id set)
            set_as_current: If True, sets this instance as current (default: True)
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        # Verify instance_id is set
        if not hasattr(response, 'instance_id') or not response.instance_id:
            raise ValueError(
                f"Response must have instance_id set. Model: {response.model_name}, "
                f"Question: {response.question_id}. Use ResultsManager._generate_instance_id() to create one."
            )

        # Create model directory if it doesn't exist
        model_dir = self.current_run_dir / "responses" / self._sanitize_name(response.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create question directory for instances
        question_dir = model_dir / response.question_id
        question_dir.mkdir(exist_ok=True)

        # Save instance response
        instance_filename = f"{self._sanitize_instance_id(response.instance_id)}.json"
        response_path = question_dir / instance_filename
        response_data = response.model_dump()

        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)

        # Update current.json pointer if requested
        if set_as_current:
            current_path = question_dir / "current.json"
            with open(current_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'instance_id': response.instance_id,
                    'last_updated': datetime.utcnow().isoformat()
                }, f, indent=2)

    def save_evaluation(self, evaluation: Evaluation):
        """
        Save an evaluation result for a specific instance.
        Supports multiple evaluation types per instance.

        Args:
            evaluation: The Evaluation to save (must have instance_id set)
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        # Verify instance_id is set
        if not hasattr(evaluation, 'instance_id') or not evaluation.instance_id:
            raise ValueError(
                f"Evaluation must have instance_id set. Model: {evaluation.model_name}, "
                f"Question: {evaluation.question_id}. Ensure the evaluation is created from a response with instance_id."
            )

        # Create evaluation directory structure: evaluations/{model}/{question_id}/{instance_id}/
        model_dir = self.current_run_dir / "evaluations" / self._sanitize_name(evaluation.model_name)
        question_dir = model_dir / evaluation.question_id
        instance_dir = question_dir / self._sanitize_instance_id(evaluation.instance_id)
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation by type (e.g., llm_judge.json, code_execution.json)
        eval_filename = f"{evaluation.evaluation_type}.json"
        eval_path = instance_dir / eval_filename
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation.model_dump(), f, indent=2, ensure_ascii=False)

    def calculate_and_save_scores(self, questions: List):
        """
        Calculate aggregate scores from current instances and save to scores.json.
        Uses the 'current' instance for each question to calculate leaderboard.
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call create_run() first.")

        responses_dir = self.current_run_dir / "responses"
        leaderboard: Dict[str, Dict] = {}

        # Iterate through all models in responses directory
        if not responses_dir.exists():
            return {}

        for model_dir in responses_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            evaluations = []
            metrics = []

            # Iterate through all question directories for this model
            for question_dir in model_dir.iterdir():
                if not question_dir.is_dir():
                    continue

                question_id = question_dir.name

                try:
                    # Get current instance ID for this question
                    current_instance_id = self.get_current_instance(
                        self.current_run.run_id, model_name, question_id
                    )

                    if not current_instance_id:
                        # No current instance found - skip this question for scoring
                        # This can happen if the question directory exists but has no valid instances
                        logger.warning(
                            "No current instance found for %s/%s. Skipping question in score calculation.",
                            model_name, question_id
                        )
                        continue

                    # Load response for metrics
                    response = self.get_response(
                        self.current_run.run_id, model_name, question_id
                    )
                    if response and response.metrics:
                        metrics.append(response.metrics)

                    # Load evaluations for this instance
                    instance_evals = self.get_instance_evaluations(
                        self.current_run.run_id, model_name, question_id, current_instance_id
                    )
                    evaluations.extend(instance_evals)

                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    # Skip questions that can't be loaded
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

    def get_response(self, run_id: str, model_name: str, question_id: str, instance_id: Optional[str] = None, use_fixed: bool = False) -> Optional[ModelResponse]:
        """
        Get a specific model response with instance support.

        Args:
            run_id: The benchmark run ID
            model_name: The model name
            question_id: The question ID
            instance_id: Specific instance to load. If None, loads current instance.
            use_fixed: If True, try to load fixed version first (legacy)

        Returns:
            ModelResponse or None
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        # Check if question directory exists (instance-based structure)
        if question_dir.exists() and question_dir.is_dir():
            if instance_id:
                # Load specific instance
                instance_path = question_dir / f"{self._sanitize_instance_id(instance_id)}.json"
                if not instance_path.exists():
                    return None
            else:
                # Load current instance
                current_instance_id = self.get_current_instance(run_id, model_name, question_id)
                if not current_instance_id:
                    return None
                instance_path = question_dir / f"{self._sanitize_instance_id(current_instance_id)}.json"
                if not instance_path.exists():
                    return None

            try:
                with open(instance_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ModelResponse(**data)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                # If instance file doesn't have required fields, skip
                pass

        # Legacy: Try fixed version first if requested
        if use_fixed:
            fixed_path = base_path / f"{question_id}_fixed.json"
            if fixed_path.exists():
                with open(fixed_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add instance fields for legacy data
                    if 'instance_id' not in data:
                        data['instance_id'] = 'legacy-fixed'
                    return ModelResponse(**data)

        # Legacy: Fall back to original flat structure
        response_path = base_path / f"{question_id}.json"
        if response_path.exists():
            with open(response_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add instance fields for legacy data
                if 'instance_id' not in data:
                    data['instance_id'] = 'legacy'
                return ModelResponse(**data)

        return None

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

    def get_all_runs_leaderboard(self) -> List[Dict]:
        """
        Get expanded leaderboard showing all runs as separate entries.
        Used for 'show all runs' mode.

        Returns:
            List of expanded leaderboard entries with run identifiers
        """
        from src.schemas import ExpandedLeaderboardEntry

        preferences = self.get_leaderboard_preferences()
        all_runs = self.get_all_runs()
        expanded_entries = []

        # Group runs by model and date for sequential numbering
        model_date_runs: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        for run in all_runs:
            date_str = run.timestamp[:8]  # YYYYMMDD
            model_date_runs[run.model][date_str].append(run)

        # Process each run
        for run in all_runs:
            leaderboard = self.get_leaderboard(run.run_id)
            if not leaderboard:
                continue

            # Get the model from this run
            model_name = run.model
            date_str = run.timestamp[:8]  # YYYYMMDD

            # Determine run identifier for model name
            if run.run_label:
                # Use custom label
                display_name = f"{model_name}-{run.run_label}"
            else:
                # Use date, with sequential number if multiple runs on same date
                runs_on_date = model_date_runs[model_name][date_str]
                if len(runs_on_date) > 1:
                    # Find position of this run (sorted by full timestamp)
                    sorted_runs = sorted(runs_on_date, key=lambda r: r.timestamp)
                    position = sorted_runs.index(run) + 1
                    display_name = f"{model_name}-{date_str}-{position}"
                else:
                    display_name = f"{model_name}-{date_str}"

            # Get model's entry from leaderboard
            sanitized_model_name = self._sanitize_name(model_name)

            # Try both original and sanitized names
            if model_name in leaderboard:
                entry_data = leaderboard[model_name]
            elif sanitized_model_name in leaderboard:
                entry_data = leaderboard[sanitized_model_name]
            else:
                continue

            # Create expanded entry
            expanded_entry = ExpandedLeaderboardEntry(
                model_name=display_name,
                original_model_name=model_name,
                run_id=run.run_id,
                run_date=date_str,
                run_label=run.run_label,
                is_preferred=(model_name in preferences and preferences[model_name] == run.run_id),
                **entry_data
            )

            expanded_entries.append(expanded_entry.model_dump())

        # Sort by overall score descending
        expanded_entries.sort(key=lambda x: x['overall_score'], reverse=True)

        return expanded_entries

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

    def get_current_instance(self, run_id: str, model_name: str, question_id: str) -> Optional[str]:
        """
        Get the current instance ID for a question.
        Falls back to latest instance if no current.json exists.

        Returns:
            instance_id or None
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        if not question_dir.exists() or not question_dir.is_dir():
            return None

        # Check for current.json
        current_path = question_dir / "current.json"
        if current_path.exists():
            with open(current_path, 'r', encoding='utf-8') as f:
                current_info = json.load(f)
                return current_info.get('instance_id')

        # Fallback: Find latest instance by filename (sorted descending)
        instance_files = sorted(question_dir.glob("*.json"), reverse=True)
        for instance_file in instance_files:
            if instance_file.name in ['current.json', 'latest.json']:
                continue
            # Read the file to get instance_id
            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'instance_id' in data:
                        return data['instance_id']
            except (OSError, json.JSONDecodeError, KeyError):
                continue

        return None

    def set_current_instance(self, run_id: str, model_name: str, question_id: str, instance_id: str):
        """
        Set the current instance for a question.
        Updates current.json pointer.

        Args:
            run_id: The benchmark run ID
            model_name: The model name
            question_id: The question ID
            instance_id: The instance ID to set as current
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        if not question_dir.exists():
            raise ValueError(f"Question directory does not exist: {question_dir}")

        # Verify instance exists
        instance_file = question_dir / f"{self._sanitize_instance_id(instance_id)}.json"
        if not instance_file.exists():
            raise ValueError(f"Instance does not exist: {instance_id}")

        # Update current.json
        current_path = question_dir / "current.json"
        with open(current_path, 'w', encoding='utf-8') as f:
            json.dump({
                'instance_id': instance_id,
                'last_updated': datetime.utcnow().isoformat()
            }, f, indent=2)

    def list_instances(self, run_id: str, model_name: str, question_id: str) -> List[Dict]:
        """
        List all instances of a response for a specific question.

        Returns:
            List of instance info dicts with keys: instance_id, instance_type, timestamp,
            is_current, has_error, evaluations, metrics
        """
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id

        if not question_dir.exists() or not question_dir.is_dir():
            return []

        # Get current instance
        current_instance_id = self.get_current_instance(run_id, model_name, question_id)

        # Find all instance files
        instances = []
        for instance_file in question_dir.glob("*.json"):
            if instance_file.name in ['current.json', 'latest.json']:
                continue

            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Skip if no instance_id (legacy files)
                    if 'instance_id' not in data:
                        continue

                    instance_id = data['instance_id']

                    # Get evaluations for this instance
                    evaluations = {}
                    eval_dir = self.results_dir / run_id / "evaluations" / self._sanitize_name(model_name) / question_id / self._sanitize_instance_id(instance_id)
                    if eval_dir.exists():
                        for eval_file in eval_dir.glob("*.json"):
                            eval_type = eval_file.stem
                            with open(eval_file, 'r', encoding='utf-8') as ef:
                                eval_data = json.load(ef)
                                evaluations[eval_type] = {
                                    'score': eval_data.get('score'),
                                    'passed': eval_data.get('passed'),
                                    'timestamp': eval_data.get('timestamp')
                                }

                    instances.append({
                        'instance_id': instance_id,
                        'instance_type': data.get('instance_type', 'original'),
                        'timestamp': data.get('timestamp'),
                        'is_current': instance_id == current_instance_id,
                        'has_error': data.get('error') is not None,
                        'error': data.get('error'),
                        'evaluations': evaluations,
                        'metrics': data.get('metrics', {})
                    })
            except (OSError, json.JSONDecodeError, KeyError):
                # Skip files that can't be loaded
                continue

        # Sort by instance_id (timestamp) descending
        instances.sort(key=lambda x: x['instance_id'], reverse=True)
        return instances

    def delete_instance(self, run_id: str, model_name: str, question_id: str, instance_id: str):
        """
        Delete a response instance and all its evaluations.
        Cannot delete the current instance or the last remaining instance.

        Args:
            run_id: The benchmark run ID
            model_name: The model name
            question_id: The question ID
            instance_id: The instance ID to delete
        """
        # Check if this is the current instance
        current_instance_id = self.get_current_instance(run_id, model_name, question_id)
        if instance_id == current_instance_id:
            raise ValueError("Cannot delete the current instance. Set a different instance as current first.")

        # Ensure at least one instance remains
        instances = self.list_instances(run_id, model_name, question_id)
        if len(instances) <= 1:
            raise ValueError("Cannot delete the last remaining instance. At least one instance must be kept.")

        # Delete response instance file
        base_path = self.results_dir / run_id / "responses" / self._sanitize_name(model_name)
        question_dir = base_path / question_id
        instance_file = question_dir / f"{self._sanitize_instance_id(instance_id)}.json"

        if instance_file.exists():
            instance_file.unlink()

        # Delete evaluation directory for this instance
        eval_dir = self.results_dir / run_id / "evaluations" / self._sanitize_name(model_name) / question_id / self._sanitize_instance_id(instance_id)
        if eval_dir.exists():
            import shutil
            shutil.rmtree(eval_dir)

    def get_instance_evaluations(self, run_id: str, model_name: str, question_id: str, instance_id: str) -> List[Evaluation]:
        """
        Get all evaluations for a specific instance.

        Returns:
            List of Evaluation objects
        """
        eval_dir = self.results_dir / run_id / "evaluations" / self._sanitize_name(model_name) / question_id / self._sanitize_instance_id(instance_id)

        if not eval_dir.exists():
            return []

        evaluations = []
        for eval_file in eval_dir.glob("*.json"):
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                evaluations.append(Evaluation(**data))

        return evaluations

    def set_run_label(self, run_id: str, label: str):
        """
        Set a custom label for a run.
        Updates the metadata.json file.

        Args:
            run_id: The benchmark run ID
            label: Custom label for this run
        """
        run_dir = self.results_dir / run_id
        metadata_path = run_dir / "metadata.json"

        if not metadata_path.exists():
            raise ValueError(f"Run not found: {run_id}")

        # Load, update, and save metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['run_label'] = label

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_run_label(self, run_id: str) -> Optional[str]:
        """
        Get the custom label for a run, if any.

        Args:
            run_id: The benchmark run ID

        Returns:
            Label string or None
        """
        run = self.get_run(run_id)
        return run.run_label if run else None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize model name for use in file paths."""
        return name.replace('/', '_').replace('\\', '_').replace(':', '_')

    @staticmethod
    def _sanitize_instance_id(instance_id: str) -> str:
        """Sanitize instance_id (ISO timestamp) for use in filenames."""
        # Convert 2025-01-15T14:30:22.123456 → 20250115T143022_123456
        return instance_id.replace('-', '').replace(':', '').replace('.', '_')

    @staticmethod
    def _generate_instance_id() -> str:
        """Generate a new instance ID (ISO timestamp with microseconds)."""
        return datetime.utcnow().isoformat(timespec='microseconds')
