"""
Benchmark Job Manager
Handles benchmark job state, persistence, and management.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import deque


class JobStatus(Enum):
    """Job status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress tracking for a benchmark job"""
    current_model: Optional[str] = None
    current_model_index: int = 0
    total_models: int = 0
    current_phase: str = "initializing"  # initializing, generating_responses, evaluating, completed
    questions_completed: int = 0
    questions_total: int = 0
    models_completed: int = 0
    elapsed_seconds: float = 0.0
    # Token usage tracking (cumulative across current model)
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_reasoning_tokens: int = 0
    cumulative_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkJob:
    """Represents a benchmark job"""
    job_id: str
    status: JobStatus
    config: Dict[str, Any]
    progress: JobProgress
    logs: List[str]
    library_logs: List[str]
    run_id: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "config": self.config,
            "progress": self.progress.to_dict(),
            "logs": self.logs[-500:],  # Only return last 500 logs
            "library_logs": self.library_logs[-500:],  # Only return last 500 library logs
            "run_id": self.run_id,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class BenchmarkJobManager:
    """
    Manages benchmark jobs with persistence and thread-safe operations.
    Enforces single active job at a time.
    """

    def __init__(self, history_file: str = "viewer/job_history.json", max_logs: int = 500):
        self.history_file = Path(history_file)
        self.max_logs = max_logs
        self.current_job: Optional[BenchmarkJob] = None
        self.job_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self._load_history()

    def _load_history(self):
        """Load job history from disk"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.job_history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load job history: {e}")
                self.job_history = []
        else:
            self.job_history = []

    def _save_history(self):
        """Persist job history to disk"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.job_history, f, indent=2)
        except Exception as e:
            print(f"Error saving job history: {e}")

    def create_job(self, job_id: str, config: Dict[str, Any]) -> BenchmarkJob:
        """
        Create a new benchmark job.
        Raises ValueError if a job is already running.
        """
        with self.lock:
            if self.current_job and self.current_job.status == JobStatus.RUNNING:
                raise ValueError("A benchmark job is already running")

            # Calculate total questions and models
            total_models = len(config.get("models", []))

            progress = JobProgress(
                total_models=total_models,
                current_model_index=0,
                questions_total=0,  # Will be updated when questions are loaded
                current_phase="initializing"
            )

            job = BenchmarkJob(
                job_id=job_id,
                status=JobStatus.PENDING,
                config=config,
                progress=progress,
                logs=[],
                library_logs=[],
                started_at=datetime.now().isoformat()
            )

            self.current_job = job
            self.add_log("Benchmark job created")

            return job

    def start_job(self):
        """Mark current job as running"""
        with self.lock:
            if not self.current_job:
                raise ValueError("No job to start")

            self.current_job.status = JobStatus.RUNNING
            self.current_job.started_at = datetime.now().isoformat()
            self.add_log("Benchmark started")

    def complete_job(self, run_id: str):
        """Mark current job as completed"""
        with self.lock:
            if not self.current_job:
                return

            self.current_job.status = JobStatus.COMPLETED
            self.current_job.run_id = run_id
            self.current_job.completed_at = datetime.now().isoformat()
            self.current_job.progress.current_phase = "completed"
            self.add_log(f"Benchmark completed successfully. Run ID: {run_id}")

            # Add to history
            self._add_to_history(self.current_job)

    def fail_job(self, error: str):
        """Mark current job as failed"""
        with self.lock:
            if not self.current_job:
                return

            self.current_job.status = JobStatus.FAILED
            self.current_job.error = error
            self.current_job.completed_at = datetime.now().isoformat()
            self.add_log(f"Benchmark failed: {error}", level="error")

            # Add to history
            self._add_to_history(self.current_job)

    def cancel_job(self):
        """Cancel the current job"""
        with self.lock:
            if not self.current_job:
                raise ValueError("No job to cancel")

            if self.current_job.status != JobStatus.RUNNING:
                raise ValueError("Can only cancel running jobs")

            self.current_job.status = JobStatus.CANCELLED
            self.current_job.completed_at = datetime.now().isoformat()
            self.add_log("Benchmark cancelled by user", level="warning")

            # Add to history
            self._add_to_history(self.current_job)

    def get_current_job(self) -> Optional[Dict[str, Any]]:
        """Get current job status"""
        with self.lock:
            if not self.current_job:
                return None
            return self.current_job.to_dict()

    def update_progress(self, **kwargs):
        """Update job progress fields"""
        with self.lock:
            if not self.current_job:
                return

            for key, value in kwargs.items():
                if hasattr(self.current_job.progress, key):
                    setattr(self.current_job.progress, key, value)

    def add_log(self, message: str, level: str = "info"):
        """
        Add a log entry to the current job.

        Args:
            message: Log message
            level: Log level (info, success, warning, error)
        """
        with self.lock:
            if not self.current_job:
                return

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

            # Log level symbols
            symbols = {
                "info": "ℹ",
                "success": "✓",
                "warning": "⚠",
                "error": "✗"
            }
            symbol = symbols.get(level, "ℹ")

            log_entry = f"[{timestamp}] {symbol} {message}"
            self.current_job.logs.append(log_entry)

            # Keep only last N logs in memory
            if len(self.current_job.logs) > self.max_logs:
                self.current_job.logs = self.current_job.logs[-self.max_logs:]

    def update_or_add_log(self, message: str, level: str = "info", update_pattern: str = None):
        """
        Update the last log entry matching a pattern, or add a new one if not found.

        Args:
            message: Log message
            level: Log level (info, success, warning, error)
            update_pattern: Pattern to match in existing logs (e.g., question ID for streaming updates)
        """
        with self.lock:
            if not self.current_job:
                return

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

            # Log level symbols
            symbols = {
                "info": "ℹ",
                "success": "✓",
                "warning": "⚠",
                "error": "✗"
            }
            symbol = symbols.get(level, "ℹ")

            log_entry = f"[{timestamp}] {symbol} {message}"

            # If update_pattern provided, try to find and update existing log
            if update_pattern and self.current_job.logs:
                # Search backwards for the most recent matching log
                for i in range(len(self.current_job.logs) - 1, -1, -1):
                    if update_pattern in self.current_job.logs[i]:
                        # Update the existing log entry
                        self.current_job.logs[i] = log_entry
                        return

            # If no match found or no pattern provided, append as new log
            self.current_job.logs.append(log_entry)

            # Keep only last N logs in memory
            if len(self.current_job.logs) > self.max_logs:
                self.current_job.logs = self.current_job.logs[-self.max_logs:]

    def add_library_log(self, message: str, level: str = "debug"):
        """
        Add a library log entry to the current job.

        Args:
            message: Log message from the rotating library
            level: Log level (debug, info, warning, error)
        """
        with self.lock:
            if not self.current_job:
                return

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

            # Log level symbols
            symbols = {
                "debug": "🔍",
                "info": "ℹ",
                "warning": "⚠",
                "error": "✗"
            }
            symbol = symbols.get(level, "ℹ")

            log_entry = f"[{timestamp}] {symbol} {message}"
            self.current_job.library_logs.append(log_entry)

            # Keep only last N logs in memory
            if len(self.current_job.library_logs) > self.max_logs:
                self.current_job.library_logs = self.current_job.library_logs[-self.max_logs:]

    def _add_to_history(self, job: BenchmarkJob):
        """Add completed job to history with full logs and metadata"""
        history_entry = job.to_dict()

        # Store full logs for later inspection
        history_entry["logs"] = self.current_job.logs.copy()
        history_entry["library_logs"] = self.current_job.library_logs.copy()

        # Add useful metadata summary
        history_entry["metadata"] = {
            "models_tested": job.config.get("models", []),
            "total_models": job.progress.total_models,
            "total_questions": job.progress.questions_total,
            "questions_completed": job.progress.questions_completed,
            "models_completed": job.progress.models_completed,
            "duration_seconds": job.progress.elapsed_seconds,
            "categories": job.config.get("categories", []),
            "max_concurrent": job.config.get("max_concurrent", 10),
        }

        self.job_history.insert(0, history_entry)  # Most recent first

        # Keep only last 50 jobs
        self.job_history = self.job_history[:50]

        self._save_history()

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get job history (most recent first)"""
        with self.lock:
            return self.job_history[:limit]

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job from history"""
        with self.lock:
            for job in self.job_history:
                if job["job_id"] == job_id:
                    return job

            # Check current job
            if self.current_job and self.current_job.job_id == job_id:
                return self.current_job.to_dict()

            return None

    def is_running(self) -> bool:
        """Check if a job is currently running"""
        with self.lock:
            return self.current_job is not None and self.current_job.status == JobStatus.RUNNING

    def clear_current_job(self):
        """Clear the current job (after completion/failure)"""
        with self.lock:
            self.current_job = None


# Global singleton instance
_job_manager_instance: Optional[BenchmarkJobManager] = None
_instance_lock = threading.Lock()


def get_job_manager() -> BenchmarkJobManager:
    """Get the global job manager instance (singleton)"""
    global _job_manager_instance

    if _job_manager_instance is None:
        with _instance_lock:
            if _job_manager_instance is None:
                _job_manager_instance = BenchmarkJobManager()

    return _job_manager_instance


class JobManagerLogHandler(logging.Handler):
    """
    Custom logging handler that forwards rotating library logs to the job manager.
    """

    def __init__(self, job_manager: BenchmarkJobManager):
        super().__init__()
        self.job_manager = job_manager

    def emit(self, record):
        """Emit a log record to the job manager"""
        try:
            # Map logging levels to our level strings
            level_map = {
                logging.DEBUG: "debug",
                logging.INFO: "info",
                logging.WARNING: "warning",
                logging.ERROR: "error",
                logging.CRITICAL: "error"
            }
            level = level_map.get(record.levelno, "debug")

            # Format the message
            message = record.getMessage()

            # Add to job manager's library logs
            self.job_manager.add_library_log(message, level=level)

        except Exception:
            # Don't let logging errors crash the application
            self.handleError(record)
