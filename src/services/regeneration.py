"""
Response regeneration service.

Handles regenerating responses for questions, creating new instances and evaluating them.
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

from src.schemas import ModelResponse, Evaluation, Question
from src.results_manager import ResultsManager
from src.rotator_client import RotatingClient

logger = logging.getLogger(__name__)


async def regenerate_response(
    run_id: str,
    model_name: str,
    question_id: str,
    question: Question,
    client: RotatingClient,
    results_manager: ResultsManager,
    evaluators: dict,
    replace_current: bool = True,
    model_options: Optional[dict] = None,
    max_instances: int = 5
) -> tuple[ModelResponse, Evaluation]:
    """
    Regenerate a response for a specific question.

    Args:
        run_id: The benchmark run ID
        model_name: The model name
        question_id: The question ID
        question: The Question object
        client: RotatingClient for API calls
        results_manager: ResultsManager instance
        evaluators: Dict of evaluators (llm_judge, code_executor, etc.)
        replace_current: If True, replaces current instance (default)
        model_options: Optional model-specific options
        max_instances: Maximum number of instances allowed per question (default: 5)

    Returns:
        tuple: (new_response, evaluation)
    """
    # Get current instance to check if it has an error
    current_instance_id = results_manager.get_current_instance(run_id, model_name, question_id)
    current_response = None
    has_error = False

    if current_instance_id:
        current_response = results_manager.get_response(run_id, model_name, question_id)
        has_error = current_response.error is not None if current_response else False

    # Check instance count (enforce max limit)
    instances = results_manager.list_instances(run_id, model_name, question_id)
    if len(instances) >= max_instances and not (has_error and replace_current):
        raise ValueError(f"Maximum of {max_instances} instances reached. Delete an instance before regenerating.")

    # Generate new response
    from src.runner import BenchmarkRunner

    # Create a temporary runner instance for generation
    # We need to set up the runner with the same config as the original run
    run = results_manager.get_run(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    # Create minimal runner for response generation
    runner = BenchmarkRunner(
        client=client,
        results_dir=str(results_manager.results_dir),
        judge_model=run.judge_model or "anthropic/claude-3-5-sonnet-20241022"
    )

    # Use the existing results_manager instance instead of the one created by BenchmarkRunner
    # This ensures we maintain the same state and avoid creating duplicate instances
    runner.results_manager = results_manager

    # Set the current run context
    runner.results_manager.current_run = run
    runner.results_manager.current_run_dir = results_manager.results_dir / run_id

    # Generate new response
    new_response = await runner._generate_response(model_name, question)

    # Set instance type and replaces field
    new_response.instance_type = "regenerated"
    if has_error and replace_current:
        new_response.replaces = current_instance_id

    # Save the new response
    # If replacing current, this will become the new current
    # Otherwise, it won't be set as current (user can switch later)
    set_as_current = replace_current
    results_manager.save_response(new_response, set_as_current=set_as_current)

    # If we're replacing an errored instance, delete the old one
    if has_error and replace_current and current_instance_id:
        try:
            # Delete old instance (this also deletes its evaluations)
            results_manager.delete_instance(run_id, model_name, question_id, current_instance_id)
        except ValueError as e:
            # Old instance might have already been deleted or set as current
            logger.warning(
                "Could not delete old instance %s for %s/%s: %s",
                current_instance_id, model_name, question_id, e
            )

    # Evaluate the new response based on question type
    if question.evaluation_type == "code_execution":
        # Run code executor first
        code_eval = await evaluators['code_executor'].evaluate(question, new_response)
        results_manager.save_evaluation(code_eval)

        # Then run LLM judge with code execution results
        llm_eval = await evaluators['llm_judge'].evaluate(
            question,
            new_response,
            code_execution_result=code_eval,
            results_dir=results_manager.current_run_dir
        )
        llm_eval.evaluation_type = "llm_judge"
        results_manager.save_evaluation(llm_eval)

        evaluation = llm_eval  # Return judge evaluation as primary

    elif question.evaluation_type == "tool_calling":
        evaluation = await evaluators['tool_validator'].evaluate(question, new_response)
        results_manager.save_evaluation(evaluation)

    elif question.evaluation_type == "exact_match":
        score = 100.0 if new_response.response_text.strip() == question.expected_output.strip() else 0.0
        evaluation = Evaluation(
            question_id=question.id,
            model_name=model_name,
            score=score,
            passed=score >= 70,
            evaluation_type="exact_match",
            reasoning="Exact match comparison",
            timestamp=datetime.now().isoformat(),
            instance_id=new_response.instance_id
        )
        results_manager.save_evaluation(evaluation)

    elif question.evaluation_type == "contains":
        score = 100.0 if question.expected_output.lower() in new_response.response_text.lower() else 0.0
        evaluation = Evaluation(
            question_id=question.id,
            model_name=model_name,
            score=score,
            passed=score >= 70,
            evaluation_type="contains",
            reasoning="Contains check",
            timestamp=datetime.now().isoformat(),
            instance_id=new_response.instance_id
        )
        results_manager.save_evaluation(evaluation)

    else:
        # Default to LLM judge
        evaluation = await evaluators['llm_judge'].evaluate(
            question,
            new_response,
            results_dir=results_manager.current_run_dir
        )
        results_manager.save_evaluation(evaluation)

    return new_response, evaluation
