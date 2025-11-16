"""
Async Benchmark Runner Wrapper
Wraps BenchmarkRunner for web execution with progress tracking.
"""

import asyncio
import logging
import sys
import threading
import time
from io import StringIO
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

from lib.rotator_library.client import RotatingClient
from src.runner import BenchmarkRunner
from viewer.benchmark_job_manager import get_job_manager, JobManagerLogHandler


class ConsoleCapture:
    """Capture console output and forward to job manager logs"""

    def __init__(self, job_manager, level="info"):
        self.job_manager = job_manager
        self.level = level
        self.buffer = StringIO()

    def write(self, text):
        """Write to buffer and job manager"""
        if text and text.strip():
            # Remove ANSI color codes for cleaner logs
            clean_text = self._strip_ansi(text.strip())
            if clean_text:
                # Detect log level from content
                level = self._detect_level(clean_text)
                self.job_manager.add_log(clean_text, level=level)

        # Also write to buffer
        self.buffer.write(text)
        return len(text)

    def flush(self):
        """Flush buffer"""
        self.buffer.flush()

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _detect_level(self, text: str) -> str:
        """Detect log level from message content"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['error', 'failed', 'exception', '✗']):
            return "error"
        elif any(word in text_lower for word in ['warning', 'warn', '⚠']):
            return "warning"
        elif any(word in text_lower for word in ['success', 'complete', '✓', '✅']):
            return "success"
        else:
            return "info"


class ProgressTracker:
    """Track benchmark progress and update job manager"""

    def __init__(self, job_manager):
        self.job_manager = job_manager
        self.start_time = time.time()

    def update(self, **kwargs):
        """Update progress with elapsed time"""
        elapsed = time.time() - self.start_time
        kwargs['elapsed_seconds'] = elapsed
        self.job_manager.update_progress(**kwargs)


class AsyncBenchmarkRunner:
    """
    Wrapper around BenchmarkRunner for async web execution.
    Captures output, tracks progress, and manages job state.
    """

    def __init__(self):
        self.job_manager = get_job_manager()
        self.cancel_event = threading.Event()
        self.runner_thread: Optional[threading.Thread] = None

    async def start_benchmark(
        self,
        job_id: str,
        client: RotatingClient,
        config: Dict[str, Any]
    ):
        """
        Start a benchmark run in the background.

        Args:
            job_id: Unique job identifier
            client: RotatingClient for API calls
            config: Benchmark configuration
        """
        # Create job
        self.job_manager.create_job(job_id, config)
        self.cancel_event.clear()

        # Start benchmark in background thread
        self.runner_thread = threading.Thread(
            target=self._run_benchmark_sync,
            args=(client, config),
            daemon=True
        )
        self.runner_thread.start()

    def _run_benchmark_sync(self, client: RotatingClient, config: Dict[str, Any]):
        """
        Run benchmark in a synchronous context (thread).
        This is needed because we need to run async code in a new event loop.
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run benchmark
            loop.run_until_complete(self._run_benchmark_async(client, config))

        except Exception as e:
            self.job_manager.fail_job(str(e))
        finally:
            loop.close()

    async def _run_benchmark_async(self, client: RotatingClient, config: Dict[str, Any]):
        """
        Actual async benchmark execution with progress tracking.
        """
        job_manager = self.job_manager
        tracker = ProgressTracker(job_manager)

        try:
            # Mark job as running
            job_manager.start_job()

            # Extract config
            models = config.get("models", [])
            categories = config.get("categories", None)
            question_ids = config.get("question_ids", None)
            max_concurrent = config.get("max_concurrent", 10)
            provider_concurrency = config.get("provider_concurrency", None)
            judge_model = config.get("judge_model", "anthropic/claude-3-5-sonnet-20241022")
            questions_dir = config.get("questions_dir", "questions")
            results_dir = config.get("results_dir", "results")

            # Get model configs
            model_configs = config.get("model_configs", {})
            model_system_instructions = {}
            model_options = {}
            model_system_instruction_positions = {}

            for model_name, model_config in model_configs.items():
                if "system_instruction" in model_config:
                    model_system_instructions[model_name] = model_config["system_instruction"]
                if "system_instruction_position" in model_config:
                    model_system_instruction_positions[model_name] = model_config["system_instruction_position"]
                if "options" in model_config:
                    model_options[model_name] = model_config["options"]

            # Get code formatting settings
            code_formatting = config.get("code_formatting_instructions", {})
            code_formatting_enabled = code_formatting.get("enabled", True)
            code_formatting_instruction = code_formatting.get("instruction", None)

            # Create benchmark runner
            runner = BenchmarkRunner(
                client=client,
                judge_model=judge_model,
                questions_dir=questions_dir,
                results_dir=results_dir,
                model_system_instructions=model_system_instructions,
                model_options=model_options,
                model_system_instruction_positions=model_system_instruction_positions,
                code_formatting_enabled=code_formatting_enabled,
                code_formatting_instruction=code_formatting_instruction
            )

            # Set up rotating library logging handler
            library_logger = logging.getLogger('rotator_library')
            library_handler = JobManagerLogHandler(job_manager)
            library_handler.setLevel(logging.DEBUG)  # Capture all levels
            library_logger.addHandler(library_handler)
            library_logger.setLevel(logging.DEBUG)
            library_logger.propagate = False  # Don't propagate to parent logger to avoid duplicates

            # Capture console output
            stdout_capture = ConsoleCapture(job_manager, level="info")
            stderr_capture = ConsoleCapture(job_manager, level="error")

            job_manager.add_log("Loading questions...", level="info")

            # Load questions to get total count
            if question_ids:
                questions = [runner.question_loader.get_question(qid) for qid in question_ids]
                questions = [q for q in questions if q is not None]
            elif categories:
                questions = []
                for category in categories:
                    questions.extend(runner.question_loader.load_category(category))
            else:
                questions = runner.question_loader.load_all_questions()

            questions_total = len(questions)
            tracker.update(
                questions_total=questions_total,
                total_models=len(models)
            )

            job_manager.add_log(
                f"Loaded {questions_total} questions across {len(models)} model(s)",
                level="success"
            )

            # Intercept runner methods to track progress
            original_run_model = runner._run_model_benchmark

            async def tracked_run_model(model: str, questions: List, max_concurrent: int):
                """Wrapped method to track per-model progress"""
                # Update current model
                current_index = models.index(model)
                tracker.update(
                    current_model=model,
                    current_model_index=current_index,
                    current_phase="generating_responses",
                    questions_completed=0
                )

                job_manager.add_log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", level="info")
                job_manager.add_log(f"Model {current_index + 1}/{len(models)}: {model}", level="info")
                job_manager.add_log(f"Phase: Generating responses for {questions_total} questions", level="info")

                # Intercept generate and evaluate methods
                original_generate = runner._generate_response_with_semaphore
                original_evaluate = runner._evaluate_response_with_semaphore

                questions_completed = 0
                evaluations_completed = 0
                evaluation_phase_started = False

                async def tracked_generate(*args, **kwargs):
                    """Track question completion during generation"""
                    nonlocal questions_completed

                    # Get question before entering semaphore
                    question = args[2] if len(args) > 2 else None

                    # We need to wrap the INNER call to log after semaphore acquisition
                    semaphore = args[0] if len(args) > 0 else None

                    # Track timing
                    start_time = time.time()

                    # Intercept to log after semaphore is acquired
                    if semaphore and question:
                        # Manually acquire semaphore first
                        async with semaphore:
                            # Now we're inside the semaphore - log it
                            job_manager.add_log(
                                f"⟳ Starting: {question.id}",
                                level="info"
                            )

                            # Define streaming progress callback
                            def stream_progress(data):
                                """Called periodically during response streaming"""
                                tokens = data.get('tokens_so_far', 0)
                                reasoning_tokens = data.get('reasoning_tokens', 0)
                                elapsed = data.get('elapsed', 0)

                                # Create progress bar (estimate 1500 tokens as target)
                                estimated_total = 1500
                                progress_pct = min(1.0, tokens / estimated_total)
                                bar_length = 10
                                filled = int(bar_length * progress_pct)
                                bar = '▓' * filled + '░' * (bar_length - filled)

                                # Calculate TPS (tokens per second)
                                total_tokens = tokens + reasoning_tokens
                                tps = total_tokens / elapsed if elapsed > 0 else 0

                                # Build details string
                                details = f"{tokens} tokens"
                                if reasoning_tokens > 0:
                                    details += f", {reasoning_tokens} reasoning"
                                details += f", {tps:.1f} TPS, {elapsed:.1f}s"

                                # Update log (replaces previous progress line for this question)
                                job_manager.update_or_add_log(
                                    f"⟳ {question.id}: {bar} Generating... ({details})",
                                    level="info",
                                    update_pattern=f"⟳ {question.id}:"
                                )

                            # Call the actual generation (without semaphore since we already have it)
                            # We need to call the inner method directly
                            model = args[1] if len(args) > 1 else None
                            progress = args[3] if len(args) > 3 else None
                            task_id = args[4] if len(args) > 4 else None

                            result = await runner._generate_response_with_retry(model, question, progress_callback=stream_progress, cancel_event=self.cancel_event)
                            if progress and task_id is not None:
                                progress.update(task_id, advance=1)
                    else:
                        # Fallback to original if we can't intercept properly
                        result = await original_generate(*args, **kwargs)

                    # Calculate elapsed time
                    elapsed = time.time() - start_time

                    questions_completed += 1

                    # Accumulate token usage
                    if result.metrics and not result.error:
                        prompt_tokens = result.metrics.get('prompt_tokens', 0)
                        completion_tokens = result.metrics.get('completion_tokens', 0)
                        reasoning_tokens = result.metrics.get('reasoning_tokens', 0)
                        cost = result.metrics.get('estimated_cost', 0)

                        # Get current cumulative values from job manager
                        current_job = job_manager.get_current_job()
                        if current_job:
                            current_progress = current_job['progress']
                            tracker.update(
                                questions_completed=questions_completed,
                                current_phase="generating_responses",
                                cumulative_prompt_tokens=current_progress['cumulative_prompt_tokens'] + prompt_tokens,
                                cumulative_completion_tokens=current_progress['cumulative_completion_tokens'] + completion_tokens,
                                cumulative_reasoning_tokens=current_progress['cumulative_reasoning_tokens'] + reasoning_tokens,
                                cumulative_cost=current_progress['cumulative_cost'] + cost
                            )
                        else:
                            tracker.update(
                                questions_completed=questions_completed,
                                current_phase="generating_responses"
                            )
                    else:
                        tracker.update(
                            questions_completed=questions_completed,
                            current_phase="generating_responses"
                        )

                    # Log AFTER completion
                    if question:
                        if result.error:
                            # Show empty progress bar for errors
                            bar = '░' * 10

                            # Update the existing progress line to show error
                            job_manager.update_or_add_log(
                                f"⟳ {question.id}: {bar} Error: {result.error}",
                                level="error",
                                update_pattern=f"⟳ {question.id}:"
                            )
                        else:
                            # Show more detail about the response
                            ttft = result.metrics.get('time_to_first_token', 0) if result.metrics else 0
                            tokens = result.metrics.get('completion_tokens', 0) if result.metrics else 0
                            reasoning_tokens = result.metrics.get('reasoning_tokens', 0) if result.metrics else 0

                            # Create full progress bar (10/10 blocks) to show completion
                            bar = '▓' * 10

                            # Calculate TPS
                            total_tokens = tokens + reasoning_tokens
                            tps = total_tokens / elapsed if elapsed > 0 else 0

                            # Build details string in same format as progress updates
                            details = f"{tokens} tokens"
                            if reasoning_tokens > 0:
                                details += f", {reasoning_tokens} reasoning"
                            details += f", {tps:.1f} TPS, {elapsed:.1f}s"

                            # Update the existing progress line to show completion
                            job_manager.update_or_add_log(
                                f"⟳ {question.id}: {bar} Completed ({details})",
                                level="success",
                                update_pattern=f"⟳ {question.id}:"
                            )

                    # Check for cancellation
                    if self.cancel_event.is_set():
                        raise Exception("Benchmark cancelled by user")

                    return result

                async def tracked_evaluate(*args, **kwargs):
                    """Track evaluation phase"""
                    nonlocal evaluations_completed, evaluation_phase_started

                    # Get parameters
                    semaphore = args[0] if len(args) > 0 else None
                    question = args[1] if len(args) > 1 else None
                    response = args[2] if len(args) > 2 else None
                    progress = args[3] if len(args) > 3 else None
                    task_id = args[4] if len(args) > 4 else None

                    # Intercept to log after semaphore is acquired
                    if semaphore and question:
                        # Manually acquire semaphore first
                        async with semaphore:
                            # Log phase transition on first evaluation
                            if not evaluation_phase_started:
                                evaluation_phase_started = True
                                job_manager.add_log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", level="info")
                                job_manager.add_log(f"Phase: Evaluating {questions_total} responses", level="info")

                            # Now we're inside the semaphore - log it
                            job_manager.add_log(
                                f"⟳ Evaluating: {question.id}",
                                level="info"
                            )

                            # Call the actual evaluation (without semaphore since we already have it)
                            result = await runner._evaluate_response(question, response)
                            if progress and task_id is not None:
                                progress.update(task_id, advance=1)
                    else:
                        # Fallback to original if we can't intercept properly
                        result = await original_evaluate(*args, **kwargs)

                    evaluations_completed += 1

                    # Update phase to evaluation
                    tracker.update(
                        current_phase="evaluating",
                        questions_completed=evaluations_completed
                    )

                    # Log evaluation result
                    if question and result:
                        score = result.score if hasattr(result, 'score') else 'N/A'
                        passed = result.passed if hasattr(result, 'passed') else False
                        status_icon = "✓" if passed else "✗"
                        job_manager.add_log(
                            f"[{evaluations_completed}/{questions_total}] {status_icon} {question.id} - Evaluated (Score: {score})",
                            level="success" if passed else "warning"
                        )

                    # Check for cancellation
                    if self.cancel_event.is_set():
                        raise Exception("Benchmark cancelled by user")

                    return result

                # Monkey patch methods
                runner._generate_response_with_semaphore = tracked_generate
                runner._evaluate_response_with_semaphore = tracked_evaluate

                # Run original method
                result = await original_run_model(model, questions, max_concurrent)

                # Restore original methods
                runner._generate_response_with_semaphore = original_generate
                runner._evaluate_response_with_semaphore = original_evaluate

                # Get final token stats before marking complete
                current_job = job_manager.get_current_job()
                if current_job:
                    current_progress = current_job['progress']

                    # Mark model as completed and reset token counters for next model
                    tracker.update(
                        models_completed=current_index + 1
                    )

                    # Log model completion summary with token breakdown
                    total_tokens = (current_progress['cumulative_prompt_tokens'] +
                                   current_progress['cumulative_completion_tokens'] +
                                   current_progress['cumulative_reasoning_tokens'])

                    summary_parts = [
                        f"✓ Model '{model}' completed",
                        f"   ↳ Tokens: {total_tokens:,} ({current_progress['cumulative_prompt_tokens']:,} prompt"
                    ]

                    if current_progress['cumulative_completion_tokens'] > 0:
                        summary_parts.append(f" • {current_progress['cumulative_completion_tokens']:,} completion")
                    if current_progress['cumulative_reasoning_tokens'] > 0:
                        summary_parts.append(f" • {current_progress['cumulative_reasoning_tokens']:,} reasoning")

                    summary_parts.append(")")

                    if current_progress['cumulative_cost'] > 0:
                        summary_parts.append(f"   ↳ Cost: ${current_progress['cumulative_cost']:.3f}")

                    job_manager.add_log(''.join(summary_parts), level="success")

                    # Reset token counters for next model
                    tracker.update(
                        cumulative_prompt_tokens=0,
                        cumulative_completion_tokens=0,
                        cumulative_reasoning_tokens=0,
                        cumulative_cost=0.0
                    )
                else:
                    # Just mark model as completed if no job found
                    tracker.update(
                        models_completed=current_index + 1
                    )

                return result

            # Monkey patch run_model method
            runner._run_model_benchmark = tracked_run_model

            # Create a timer update task that runs every second
            async def update_timer():
                """Update elapsed time every second independently"""
                while not self.cancel_event.is_set():
                    try:
                        await asyncio.sleep(1)
                        tracker.update()  # Just update elapsed time
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        # Continue even if there's an error
                        continue

            # Start timer task
            timer_task = asyncio.create_task(update_timer())

            try:
                # Run benchmark with console capture
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    run_id = await runner.run_benchmark(
                        models=models,
                        categories=categories,
                        question_ids=question_ids,
                        max_concurrent=max_concurrent,
                        provider_concurrency=provider_concurrency
                    )

                # Mark as completed
                job_manager.complete_job(run_id)
            finally:
                # Cancel timer task
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass

                # Clean up logging handler
                library_logger.removeHandler(library_handler)

        except Exception as e:
            error_msg = str(e)

            # Check if it was a cancellation
            if "cancelled" in error_msg.lower():
                job_manager.cancel_job()
            else:
                job_manager.fail_job(error_msg)

    def cancel(self):
        """Cancel the running benchmark"""
        if self.job_manager.is_running():
            self.cancel_event.set()
            job_manager = self.job_manager
            job_manager.add_log("Cancellation requested...", level="warning")

            # Wait for thread to finish (with timeout)
            if self.runner_thread and self.runner_thread.is_alive():
                self.runner_thread.join(timeout=15.0)
        else:
            raise ValueError("No benchmark is currently running")

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current job status"""
        return self.job_manager.get_current_job()


# Global singleton instance
_runner_instance: Optional[AsyncBenchmarkRunner] = None
_instance_lock = threading.Lock()


def get_async_runner() -> AsyncBenchmarkRunner:
    """Get the global async runner instance (singleton)"""
    global _runner_instance

    if _runner_instance is None:
        with _instance_lock:
            if _runner_instance is None:
                _runner_instance = AsyncBenchmarkRunner()

    return _runner_instance
