"""
Main benchmark runner for executing benchmarks and collecting results.
"""
import asyncio
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lib.rotator_library.client import RotatingClient
from src.question_loader import QuestionLoader
from src.results_manager import ResultsManager
from src.schemas import Question, ModelResponse, Evaluation
from src.cost_calculator import CostCalculator
from src.evaluators import LLMJudgeEvaluator, ToolCallValidator, CodeExecutor
from src.reasoning_extractor import ReasoningExtractor


class BenchmarkRunner:
    """Main benchmark runner orchestrating the entire process."""

    def __init__(
        self,
        client: RotatingClient,
        judge_model: str = "anthropic/claude-3-5-sonnet-20241022",
        questions_dir: str = "questions",
        results_dir: str = "results",
        model_system_instructions: Optional[Dict[str, str]] = None,
        code_formatting_enabled: bool = True,
        code_formatting_instruction: Optional[str] = None
    ):
        self.client = client
        self.judge_model = judge_model
        self.console = Console()
        self.model_system_instructions = model_system_instructions or {}
        self.code_formatting_enabled = code_formatting_enabled
        self.code_formatting_instruction = code_formatting_instruction

        self.question_loader = QuestionLoader(questions_dir)
        self.results_manager = ResultsManager(results_dir)

        # Initialize evaluators
        self.llm_judge = LLMJudgeEvaluator(client, judge_model)
        self.tool_validator = ToolCallValidator()
        self.code_executor = CodeExecutor()

    async def run_benchmark(
        self,
        models: List[str],
        categories: Optional[List[str]] = None,
        question_ids: Optional[List[str]] = None,
        max_concurrent: int = 3
    ) -> str:
        """
        Run a complete benchmark.

        Args:
            models: List of model names to benchmark
            categories: Optional list of categories to include
            question_ids: Optional list of specific question IDs to run
            max_concurrent: Maximum number of concurrent requests

        Returns:
            str: The run_id for the completed benchmark
        """
        self.console.print("[bold red]🔴 MirroBench - Starting Benchmark[/bold red]")
        self.console.print(f"Models: {', '.join(models)}")
        self.console.print(f"Judge Model: {self.judge_model}\n")

        # Load questions
        self.console.print("[yellow]Loading questions...[/yellow]")
        if question_ids:
            questions = [self.question_loader.get_question(qid) for qid in question_ids]
            questions = [q for q in questions if q is not None]
        elif categories:
            questions = []
            for category in categories:
                questions.extend(self.question_loader.load_category(category))
        else:
            questions = self.question_loader.load_all_questions()

        if not questions:
            self.console.print("[red]No questions found![/red]")
            return ""

        self.console.print(f"[green]Loaded {len(questions)} questions[/green]\n")

        # Run benchmarks for each model (each gets its own run)
        run_ids = []
        for model in models:
            self.console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            self.console.print(f"[bold cyan]Starting Run for: {model}[/bold cyan]")
            self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

            # Create run for this model
            run_id = self.results_manager.create_run(
                model=model,
                categories=categories or self.question_loader.get_categories(),
                total_questions=len(questions),
                judge_model=self.judge_model,
                config={
                    "max_concurrent": max_concurrent,
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.console.print(f"[bold green]Run ID: {run_id}[/bold green]\n")
            run_ids.append(run_id)

            # Run benchmark for this model
            await self._run_model_benchmark(model, questions, max_concurrent)

            # Calculate and save scores for this run
            self.console.print("\n[yellow]Calculating scores...[/yellow]")
            leaderboard = self.results_manager.calculate_and_save_scores(questions)

            # Display model summary
            if model in leaderboard:
                entry = leaderboard[model]
                self.console.print(f"\n[bold]Results for {model}:[/bold]")
                self.console.print(f"  Score: {entry['overall_score']:.1f}/100")
                self.console.print(f"  Passed: {entry['passed_questions']}/{entry['total_questions']}")
                if entry.get('total_cost'):
                    self.console.print(f"  Cost: ${entry['total_cost']:.4f}\n")

        # Display final summary
        self.console.print(f"\n[bold green]✓ Benchmark complete![/bold green]")
        self.console.print(f"[green]Completed {len(models)} model(s) with {len(questions)} questions each[/green]")
        self.console.print(f"[red]View results: python viewer/server.py[/red]\n")

        # Return the last run_id (or all if needed)
        return run_ids[-1] if run_ids else ""

    async def _run_model_benchmark(
        self,
        model: str,
        questions: List[Question],
        max_concurrent: int
    ):
        """Run benchmark for a single model."""
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Phase 1: Generate responses
            response_task = progress.add_task(
                f"[cyan]Generating responses...",
                total=len(questions)
            )

            responses = []
            tasks = []

            for question in questions:
                task = self._generate_response_with_semaphore(
                    semaphore, model, question, progress, response_task
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            # Phase 2: Evaluate responses
            eval_task = progress.add_task(
                f"[yellow]Evaluating responses...",
                total=len(questions)
            )

            eval_tasks = []
            for question, response in zip(questions, responses):
                task = self._evaluate_response_with_semaphore(
                    semaphore, question, response, progress, eval_task
                )
                eval_tasks.append(task)

            evaluations = await asyncio.gather(*eval_tasks)

    async def _generate_response_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        model: str,
        question: Question,
        progress: Progress,
        task_id: int
    ) -> ModelResponse:
        """Generate a response with concurrency control and retry logic."""
        async with semaphore:
            response = await self._generate_response_with_retry(model, question)
            progress.update(task_id, advance=1)
            return response

    async def _generate_response_with_retry(
        self,
        model: str,
        question: Question,
        max_retries: int = 3
    ) -> ModelResponse:
        """Generate a response with retry logic for failures."""
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self._generate_response(model, question)

                # Check if response has an error
                if response.error:
                    last_error = response.error
                    self.console.print(f"[yellow]⚠️  Attempt {attempt + 1}/{max_retries} failed for {question.id}: {response.error}[/yellow]")

                    # If not the last attempt, retry after delay
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        self.console.print(f"[yellow]   Retrying in {delay}s...[/yellow]")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Success!
                    if attempt > 0:
                        self.console.print(f"[green]✓ Retry successful for {question.id} on attempt {attempt + 1}[/green]")
                    return response

            except Exception as e:
                last_error = str(e)
                self.console.print(f"[yellow]⚠️  Attempt {attempt + 1}/{max_retries} failed for {question.id}: {str(e)}[/yellow]")

                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    self.console.print(f"[yellow]   Retrying in {delay}s...[/yellow]")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, create error response
                    return ModelResponse(
                        question_id=question.id,
                        model_name=model,
                        response_text="",
                        reasoning_content=None,
                        tool_calls=None,
                        full_response={},
                        metrics={},
                        timestamp=datetime.now().isoformat(),
                        error=f"All {max_retries} attempts failed. Last error: {last_error}"
                    )

        # This should not be reached, but just in case
        return response if 'response' in locals() else ModelResponse(
            question_id=question.id,
            model_name=model,
            response_text="",
            reasoning_content=None,
            tool_calls=None,
            full_response={},
            metrics={},
            timestamp=datetime.now().isoformat(),
            error=f"All retries exhausted. Last error: {last_error}"
        )

    async def _generate_response(self, model: str, question: Question) -> ModelResponse:
        """Generate a response from a model for a question."""
        start_time = time.time()
        first_token_time = None
        response_text = ""
        reasoning_content = ""
        tool_calls = None
        full_response_obj = None
        error = None

        try:
            # Build messages
            messages = []

            # Build system prompt (combine model-specific instructions with question system prompt)
            system_content_parts = []

            # Add model-specific system instructions if configured
            if model in self.model_system_instructions:
                system_content_parts.append(self.model_system_instructions[model])

            # Add code formatting instructions for code-based questions
            if self.code_formatting_enabled and self.code_formatting_instruction:
                # Check if this is a code-based question
                is_code_question = (
                    question.evaluation_type in ["code_execution", "code_execution_multi_file"] or
                    question.category in ["games", "web_apps", "visualizations", "simulations", "creative_coding", "cli_tools"] or
                    "code" in question.tags or
                    "coding" in question.tags
                )
                if is_code_question:
                    system_content_parts.append(self.code_formatting_instruction)

            # Add question-specific system prompt
            if question.system_prompt:
                system_content_parts.append(question.system_prompt)

            # Add combined system message if we have any system content
            if system_content_parts:
                messages.append({
                    "role": "system",
                    "content": "\n\n".join(system_content_parts)
                })

            messages.append({"role": "user", "content": question.prompt})

            # Add tools if needed
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": True
            }

            if question.tools:
                # Convert tools to proper format
                tools = []
                for tool_def in question.tools:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": tool_def.parameters
                        }
                    })
                kwargs["tools"] = tools

            # Stream response
            response_stream = self.client.acompletion(**kwargs)
            stream_iterator = response_stream.__aiter__()

            all_chunks = []

            async for chunk_str in stream_iterator:
                if not chunk_str.startswith("data: "):
                    continue

                data_str = chunk_str[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data_str)
                    all_chunks.append(chunk_data)

                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})

                        # Track first token
                        if first_token_time is None and (delta.get("content") or delta.get("reasoning_content")):
                            first_token_time = time.time()

                        if delta.get("content"):
                            response_text += delta["content"]

                        if delta.get("reasoning_content"):
                            reasoning_content += delta["reasoning_content"]

                        if delta.get("tool_calls"):
                            if tool_calls is None:
                                tool_calls = []
                            tool_calls.extend(delta["tool_calls"])

                except json.JSONDecodeError:
                    continue

            end_time = time.time()

            # Extract reasoning from response text if present (e.g., <think>...</think>)
            cleaned_response_text = response_text
            extracted_reasoning = None
            reasoning_format = None

            if response_text and not reasoning_content:
                # Only extract if reasoning_content wasn't already set by API
                cleaned_response_text, extracted_reasoning, reasoning_format = ReasoningExtractor.extract_reasoning(response_text)

                if extracted_reasoning:
                    # Use extracted reasoning as the reasoning_content
                    reasoning_content = extracted_reasoning

            # Reassemble full response
            if all_chunks:
                full_response_obj = all_chunks[0].copy()
                usage_data = None

                for chunk in all_chunks:
                    if "usage" in chunk:
                        usage_data = chunk["usage"]

                # Finalize response
                if "choices" in full_response_obj and full_response_obj["choices"]:
                    full_response_obj["choices"][0].pop("delta", None)
                    final_message = {
                        "role": "assistant",
                        "content": cleaned_response_text  # Use cleaned text without reasoning tags
                    }
                    if reasoning_content:
                        final_message["reasoning_content"] = reasoning_content
                    if tool_calls:
                        final_message["tool_calls"] = tool_calls

                    full_response_obj["choices"][0]["message"] = final_message

                if usage_data:
                    full_response_obj["usage"] = usage_data

            # Calculate metrics
            ttft = (first_token_time - start_time) if first_token_time else 0
            total_latency = end_time - start_time
            generation_time = (end_time - first_token_time) if first_token_time else total_latency

            prompt_tokens = usage_data.get("prompt_tokens", 0) if usage_data else 0
            completion_tokens = usage_data.get("completion_tokens", 0) if usage_data else 0
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens) if usage_data else 0

            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

            # Calculate cost
            estimated_cost = CostCalculator.calculate_cost(
                model, prompt_tokens, completion_tokens
            )

            metrics = {
                "ttft": ttft,
                "total_latency": total_latency,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost
            }

        except Exception as e:
            error = str(e)
            metrics = {}

        # Create response object
        model_response = ModelResponse(
            question_id=question.id,
            model_name=model,
            response_text=cleaned_response_text,  # Use cleaned text without reasoning tags
            reasoning_content=reasoning_content if reasoning_content else None,
            tool_calls=tool_calls,
            full_response=full_response_obj or {},
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            error=error
        )

        # Save response
        self.results_manager.save_response(model_response)

        return model_response

    async def _evaluate_response_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        question: Question,
        response: ModelResponse,
        progress: Progress,
        task_id: int
    ) -> Evaluation:
        """Evaluate a response with concurrency control and retry logic."""
        async with semaphore:
            evaluation = await self._evaluate_response_with_retry(question, response)
            progress.update(task_id, advance=1)
            return evaluation

    async def _evaluate_response_with_retry(
        self,
        question: Question,
        response: ModelResponse,
        max_retries: int = 3
    ) -> Evaluation:
        """Evaluate a response with retry logic for failures."""
        last_error = None

        for attempt in range(max_retries):
            try:
                evaluation = await self._evaluate_response(question, response)

                # Check if evaluation failed (reasoning contains "failed" or score is 0 with error in details)
                is_failed = (
                    evaluation.reasoning and "Evaluation failed:" in evaluation.reasoning or
                    evaluation.details and "evaluation_error" in evaluation.details
                )

                if is_failed:
                    last_error = evaluation.reasoning
                    self.console.print(f"[yellow]⚠️  Judge attempt {attempt + 1}/{max_retries} failed for {question.id}: {evaluation.reasoning}[/yellow]")

                    if attempt < max_retries - 1:
                        delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        self.console.print(f"[yellow]   Retrying evaluation in {delay}s...[/yellow]")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Success!
                    if attempt > 0:
                        self.console.print(f"[green]✓ Evaluation retry successful for {question.id} on attempt {attempt + 1}[/green]")
                    return evaluation

            except Exception as e:
                last_error = str(e)
                self.console.print(f"[yellow]⚠️  Evaluation attempt {attempt + 1}/{max_retries} failed for {question.id}: {str(e)}[/yellow]")

                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    self.console.print(f"[yellow]   Retrying evaluation in {delay}s...[/yellow]")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, create error evaluation
                    return Evaluation(
                        question_id=question.id,
                        model_name=response.model_name,
                        score=0.0,
                        passed=False,
                        evaluation_type="llm_judge",
                        evaluator_model=self.judge_model,
                        reasoning=f"All {max_retries} evaluation attempts failed. Last error: {last_error}",
                        details={"retry_exhausted": True, "last_error": str(last_error)},
                        timestamp=datetime.now().isoformat()
                    )

        # This should not be reached, but just in case
        return evaluation if 'evaluation' in locals() else Evaluation(
            question_id=question.id,
            model_name=response.model_name,
            score=0.0,
            passed=False,
            evaluation_type="llm_judge",
            evaluator_model=self.judge_model,
            reasoning=f"All retries exhausted. Last error: {last_error}",
            details={"retry_exhausted": True},
            timestamp=datetime.now().isoformat()
        )

    async def _evaluate_response(self, question: Question, response: ModelResponse) -> Evaluation:
        """
        Evaluate a response based on the question's evaluation type.
        For code execution questions, we run BOTH code executor AND judge separately.
        """
        # For code execution questions: run BOTH code executor AND judge
        if question.evaluation_type == "code_execution":
            # Run code executor to get technical validation
            code_eval = await self.code_executor.evaluate(question, response)
            self.results_manager.save_evaluation(code_eval)

            # ALSO run LLM judge to evaluate response quality (separate from code execution)
            judge_eval = await self.llm_judge.evaluate(question, response)
            judge_eval.evaluation_type = "llm_judge"  # Mark as judge evaluation
            self.results_manager.save_evaluation(judge_eval)

            # Return judge evaluation as primary (but code execution is also saved)
            return judge_eval

        elif question.evaluation_type == "tool_calling":
            evaluation = await self.tool_validator.evaluate(question, response)
        elif question.evaluation_type == "exact_match":
            # Simple exact match evaluation
            score = 100.0 if response.response_text.strip() == question.expected_output.strip() else 0.0
            evaluation = Evaluation(
                question_id=question.id,
                model_name=response.model_name,
                score=score,
                passed=score >= 70,
                evaluation_type="exact_match",
                reasoning="Exact match comparison",
                timestamp=datetime.now().isoformat()
            )
        elif question.evaluation_type == "contains":
            # Simple contains evaluation
            score = 100.0 if question.expected_output.lower() in response.response_text.lower() else 0.0
            evaluation = Evaluation(
                question_id=question.id,
                model_name=response.model_name,
                score=score,
                passed=score >= 70,
                evaluation_type="contains",
                reasoning="Contains check",
                timestamp=datetime.now().isoformat()
            )
        else:
            # Default to LLM judge (for "llm_judge" and any other types)
            evaluation = await self.llm_judge.evaluate(question, response)

        # Save evaluation
        self.results_manager.save_evaluation(evaluation)

        return evaluation

    def _display_model_summary(self, model: str, evaluations: List[Evaluation]):
        """Display a summary of model performance."""
        passed = sum(1 for e in evaluations if e.passed)
        total = len(evaluations)
        avg_score = sum(e.score for e in evaluations) / total if total > 0 else 0

        self.console.print(f"\n[bold]Summary for {model}:[/bold]")
        self.console.print(f"  Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        self.console.print(f"  Average Score: {avg_score:.1f}/100\n")

    def _display_leaderboard(self, leaderboard: Dict[str, Any]):
        """Display the final leaderboard."""
        table = Table(title="[bold cyan]Final Leaderboard[/bold cyan]")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Model", justify="left", style="magenta")
        table.add_column("Score", justify="center", style="green")
        table.add_column("Passed", justify="center", style="yellow")
        table.add_column("Avg TPS", justify="center", style="blue")
        table.add_column("Total Cost", justify="center", style="red")

        for rank, (model_name, entry) in enumerate(leaderboard.items(), 1):
            table.add_row(
                str(rank),
                model_name,
                f"{entry['overall_score']:.1f}",
                f"{entry['passed_questions']}/{entry['total_questions']}",
                f"{entry.get('avg_tps', 0):.1f}" if entry.get('avg_tps') else "N/A",
                CostCalculator.format_cost(entry.get('total_cost', 0)) if entry.get('total_cost') else "N/A"
            )

        self.console.print(table)
