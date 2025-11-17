"""
FastAPI server for viewing benchmark results.

Usage:
    python viewer/server.py
    Then open http://localhost:8000 in your browser
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results_manager import ResultsManager
from src.question_loader import QuestionLoader
from src.artifact_extractor import ArtifactExtractor
from src.code_fixer import CodeFixer
from src.config_loader import ConfigLoader


class PreferenceRequest(BaseModel):
    """Request body for setting leaderboard preference."""
    run_id: str


class FixRequest(BaseModel):
    """Request body for fixing response formatting."""
    fixer_model: Optional[str] = None  # If not provided, uses default from config


# Define project root (parent of viewer directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Change working directory to project root so all relative paths work correctly
# This ensures oauth_creds, results, etc. are created in the project root
os.chdir(PROJECT_ROOT)

# Load configuration from project root
try:
    config = ConfigLoader(str(PROJECT_ROOT / "config.yaml"))
except:
    config = None

# Initialize artifact extractor
artifact_extractor = ArtifactExtractor()

# Initialize code fixer (will be set up with client when needed)
code_fixer = None

# Persistent temp directories per session (key: session_id, value: temp_dir_path)
# Session ID format: {runId}_{modelName}_{questionId}_{version}
persistent_temp_dirs = {}


def _extract_and_combine_code_blocks(text: str) -> Optional[str]:
    """
    Extract all code blocks from response and combine them into a single HTML artifact.
    Handles cases where HTML, CSS, and JavaScript are in separate blocks.
    """
    import re

    # Find all code blocks with their language specifiers
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    html_blocks = []
    css_blocks = []
    js_blocks = []
    python_blocks = []

    # Categorize blocks by type
    for lang, code in matches:
        lang = lang.lower() if lang else ''
        code = code.strip()

        if lang in ['html', 'htm'] or '<html' in code.lower() or '<!DOCTYPE' in code:
            html_blocks.append(code)
        elif lang in ['css']:
            css_blocks.append(code)
        elif lang in ['javascript', 'js']:
            js_blocks.append(code)
        elif lang in ['python', 'py']:
            python_blocks.append(code)
        elif not lang:
            # No language specified - try to infer
            if '<html' in code.lower() or '<!DOCTYPE' in code:
                html_blocks.append(code)
            elif code.strip().startswith('{') or 'function' in code or 'const ' in code or 'let ' in code:
                js_blocks.append(code)

    # If we have a complete HTML block, use it
    if html_blocks:
        # Check if it's a complete HTML document
        for html in html_blocks:
            if '<html' in html.lower() or '<!DOCTYPE' in html.lower():
                # Complete document - inject any separate CSS/JS
                result = html

                # Inject CSS if present and not already in HTML
                if css_blocks and '<style>' not in html.lower():
                    css_content = '\n'.join(css_blocks)
                    style_tag = f'<style>\n{css_content}\n</style>'
                    if '</head>' in html.lower():
                        result = result.replace('</head>', f'{style_tag}\n</head>')
                    else:
                        result = result.replace('<html>', f'<html>\n<head>\n{style_tag}\n</head>')

                # Inject JS if present and not already in HTML
                if js_blocks and '<script>' not in html.lower():
                    js_content = '\n'.join(js_blocks)
                    script_tag = f'<script>\n{js_content}\n</script>'
                    if '</body>' in html.lower():
                        result = result.replace('</body>', f'{script_tag}\n</body>')
                    else:
                        result = result + f'\n{script_tag}'

                return result

        # Partial HTML - wrap it
        html_content = '\n'.join(html_blocks)
        css_content = '\n'.join(css_blocks)
        js_content = '\n'.join(js_blocks)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {'<style>' + css_content + '</style>' if css_content else ''}
</head>
<body>
    {html_content}
    {'<script>' + js_content + '</script>' if js_content else ''}
</body>
</html>"""

    # Only CSS and JS - create minimal HTML wrapper
    if css_blocks or js_blocks:
        css_content = '\n'.join(css_blocks)
        js_content = '\n'.join(js_blocks)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {'<style>' + css_content + '</style>' if css_content else ''}
</head>
<body>
    <div id="app"></div>
    {'<script>' + js_content + '</script>' if js_content else ''}
</body>
</html>"""

    return None


app = FastAPI(title="MirroBench - LLM Benchmark Viewer", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers with paths relative to project root
results_dir = str(PROJECT_ROOT / (config.results_dir if config else "results"))
questions_dir = str(PROJECT_ROOT / (config.questions_dir if config else "questions"))

results_manager = ResultsManager(results_dir)
question_loader = QuestionLoader(questions_dir)

# Try to load questions
try:
    question_loader.load_all_questions()
except Exception as e:
    print(f"Warning: Could not load questions: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="<h1>LLM Benchmark Viewer</h1><p>Template not found. Please create viewer/templates/index.html</p>")


@app.get("/benchmark-log-popout", response_class=HTMLResponse)
async def benchmark_log_popout():
    """Serve the benchmark log popout window."""
    html_path = Path(__file__).parent / "templates" / "log_popout.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="<h1>Log Popout</h1><p>Template not found.</p>")


@app.get("/api/runs")
async def get_runs():
    """Get all benchmark runs."""
    try:
        runs = results_manager.get_all_runs()
        return {
            "runs": [
                {
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "model": run.model,  # Single model per run
                    "categories": run.categories,
                    "total_questions": run.total_questions,
                    "judge_model": run.judge_model
                }
                for run in runs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get details for a specific run."""
    try:
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RunLabelRequest(BaseModel):
    """Request body for setting run label."""
    label: str


@app.put("/api/runs/{run_id}/label")
async def set_run_label(run_id: str, request: RunLabelRequest):
    """Set a custom label for a run."""
    try:
        # Verify run exists
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Set the label
        results_manager.set_run_label(run_id, request.label)

        return {
            "success": True,
            "run_id": run_id,
            "label": request.label
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/label")
async def get_run_label(run_id: str):
    """Get the label for a run."""
    try:
        # Verify run exists
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        label = results_manager.get_run_label(run_id)

        return {
            "run_id": run_id,
            "label": label
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/bulk-data")
async def get_run_bulk_data(run_id: str, model_name: str):
    """
    Get all data for a run at once for client-side caching.
    Returns: run metadata, all responses, all evaluations, and questions.
    """
    try:
        # Get run metadata
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get all questions for the categories in this run
        all_questions = {}
        for question in question_loader.questions.values():
            # Include questions that match categories OR have null/empty category
            if not question.category or question.category in run.categories:
                all_questions[question.id] = question.model_dump()

        # Get all responses and evaluations for this model
        responses = {}
        evaluations = {}

        responses_dir = results_manager.results_dir / run_id / "responses" / results_manager._sanitize_name(model_name)
        if responses_dir.exists():
            # Collect question IDs from both versioned (subdirectories) and legacy (flat) structures
            question_ids = set()

            # Check for versioned structure (subdirectories with version files)
            for question_dir in responses_dir.iterdir():
                if question_dir.is_dir():
                    # This is a question directory in versioned structure
                    question_ids.add(question_dir.name)

            # Check for legacy flat structure (JSON files directly in model dir)
            for response_file in responses_dir.glob("*.json"):
                if '_fixed' not in response_file.stem:
                    question_ids.add(response_file.stem)

            # Load responses and evaluations for each question
            for question_id in question_ids:
                try:
                    # Get response (handles both versioned and legacy structures)
                    response = results_manager.get_response(run_id, model_name, question_id)
                    if response:
                        responses[question_id] = response.model_dump()

                        # Check if fixed version exists
                        has_fixed = results_manager.has_fixed_response(run_id, model_name, question_id)
                        responses[question_id]['has_fixed_version'] = has_fixed

                    # Get all evaluations for this question
                    all_evals = results_manager.get_all_evaluations(run_id, model_name, question_id)
                    if all_evals:
                        evaluations[question_id] = {}
                        for eval_obj in all_evals:
                            evaluations[question_id][eval_obj.evaluation_type] = eval_obj.model_dump()

                except Exception as e:
                    print(f"Error loading {question_id}: {e}")
                    continue

        return {
            "run": run.model_dump(),
            "questions": all_questions,
            "responses": responses,
            "evaluations": evaluations,
            "model_name": model_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/leaderboard")
async def get_leaderboard(run_id: str):
    """Get leaderboard for a run (uses latest run for each model by default)."""
    try:
        leaderboard = results_manager.get_leaderboard(run_id)
        if not leaderboard:
            raise HTTPException(status_code=404, detail="Leaderboard not found")
        return {"leaderboard": leaderboard}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}")
async def get_response(run_id: str, model_name: str, question_id: str, use_fixed: bool = False, instance_id: Optional[str] = None):
    """Get a specific response with all evaluations, supports instance_id parameter."""
    try:
        # Get the response (original, fixed, or specific instance)
        response = results_manager.get_response(run_id, model_name, question_id, use_fixed=use_fixed, instance_id=instance_id)
        if not response:
            raise HTTPException(status_code=404, detail="Response not found")

        # Check if fixed version exists
        has_fixed = results_manager.has_fixed_response(run_id, model_name, question_id)

        # Get ALL evaluations for this question
        all_evaluations = results_manager.get_all_evaluations(run_id, model_name, question_id)

        # Organize evaluations by type
        evaluations_by_type = {}
        for eval_obj in all_evaluations:
            evaluations_by_type[eval_obj.evaluation_type] = eval_obj.model_dump()

        # Get primary evaluation (for backward compatibility)
        evaluation = results_manager.get_evaluation(run_id, model_name, question_id)

        # Get the question
        question = question_loader.get_question(question_id)

        # Extract HTML/code artifact if present
        artifact = None
        artifact_type = None
        artifact_metadata = None

        if response.response_text:
            # First try multi-file extraction
            artifact_id = f"{run_id}_{model_name}_{question_id}".replace('/', '_')
            multi_file_artifact = artifact_extractor.extract_multi_file_artifact(
                response.response_text,
                artifact_id
            )

            if multi_file_artifact:
                if multi_file_artifact['type'] == 'multi_file':
                    # Multi-file web app
                    artifact_type = 'multi_file'
                    artifact_metadata = {
                        'entry_point': multi_file_artifact['entry_point'],
                        'file_count': multi_file_artifact['file_count'],
                        'files': list(multi_file_artifact['files'].keys()),
                        'artifact_id': artifact_id
                    }
                    # For multi-file, provide URL to static directory
                    artifact = f"/artifacts/{artifact_id}/{multi_file_artifact['entry_point']}"
                else:
                    # Single file
                    artifact = multi_file_artifact['content']
                    artifact_type = 'html'
            else:
                # Fallback to old single-file extraction
                if '<!DOCTYPE' in response.response_text or '<html' in response.response_text.lower():
                    if '```' not in response.response_text:
                        artifact = response.response_text
                        artifact_type = 'html'
                    else:
                        artifact = _extract_and_combine_code_blocks(response.response_text)
                        if artifact:
                            artifact_type = 'html'
                elif '```' in response.response_text:
                    artifact = _extract_and_combine_code_blocks(response.response_text)
                    if artifact:
                        artifact_type = 'html'

        return {
            "question": question.model_dump() if question else None,
            "response": response.model_dump(),
            "evaluation": evaluation.model_dump() if evaluation else None,
            "evaluations": evaluations_by_type,  # All evaluations organized by type
            "artifact": artifact,
            "artifact_type": artifact_type,
            "artifact_metadata": artifact_metadata,
            "has_fixed_version": has_fixed,
            "is_fixed_version": use_fixed
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/re-evaluate")
async def re_evaluate_response(run_id: str, model_name: str, question_id: str):
    """Re-evaluate a response (runs both judge and code execution if applicable)."""
    try:
        # Get original response
        response = results_manager.get_response(run_id, model_name, question_id)
        if not response:
            raise HTTPException(status_code=404, detail="Response not found")

        # Get question
        question = question_loader.get_question(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Set up the run directory for saving evaluations
        results_manager.current_run_dir = results_manager.results_dir / run_id

        # Initialize evaluators
        from src.rotator_client import RotatingClient

        # Collect API keys
        api_keys = defaultdict(list)
        for key, value in os.environ.items():
            if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
                parts = key.split("_API_KEY")
                provider = parts[0].lower()
                if provider not in api_keys:
                    api_keys[provider] = []
                api_keys[provider].append(value)

        if not api_keys:
            raise HTTPException(status_code=500, detail="No API keys configured for evaluation")

        # Create client with retry settings from config if available
        client_kwargs = {"api_keys": dict(api_keys)}
        if config:
            client_kwargs["max_retries"] = config.max_retries_per_key
            client_kwargs["global_timeout"] = config.global_timeout
            client_kwargs["max_concurrent_requests_per_key"] = config.provider_concurrency
        client = RotatingClient(**client_kwargs)

        # Get judge model from config or use default
        if config and hasattr(config, 'judge_model'):
            judge_model = config.judge_model
        else:
            judge_model = "anthropic/claude-3-5-sonnet-20241022"

        # Import evaluators
        from src.evaluators import LLMJudgeEvaluator, CodeExecutor

        llm_judge = LLMJudgeEvaluator(client, judge_model)
        code_executor = CodeExecutor()

        # Run evaluations based on question type
        evaluations_run = []
        code_eval = None

        # For code execution questions: run code executor FIRST, then pass results to LLM judge
        if question.evaluation_type == "code_execution":
            code_eval = await code_executor.evaluate(question, response)
            results_manager.save_evaluation(code_eval)
            evaluations_run.append("code_execution")

        # Always run LLM judge (with code execution results if available)
        judge_eval = await llm_judge.evaluate(
            question,
            response,
            code_execution_result=code_eval,
            results_dir=results_manager.current_run_dir
        )
        judge_eval.evaluation_type = "llm_judge"
        results_manager.save_evaluation(judge_eval)
        evaluations_run.append("llm_judge")

        return {
            "success": True,
            "message": f"Re-evaluation completed. Evaluations run: {', '.join(evaluations_run)}",
            "evaluations_run": evaluations_run
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-evaluation failed: {str(e)}")


@app.post("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/fix")
async def fix_response_formatting(run_id: str, model_name: str, question_id: str, request: FixRequest):
    """Fix response formatting using LLM."""
    try:
        global code_fixer

        # Initialize code fixer if needed (lazy initialization)
        if code_fixer is None:
            from src.rotator_client import RotatingClient

            # Collect API keys
            api_keys = defaultdict(list)
            for key, value in os.environ.items():
                if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
                    parts = key.split("_API_KEY")
                    provider = parts[0].lower()
                    if provider not in api_keys:
                        api_keys[provider] = []
                    api_keys[provider].append(value)

            if not api_keys:
                raise HTTPException(status_code=500, detail="No API keys configured for code fixer")

            # Create client with retry settings from config if available
            client_kwargs = {"api_keys": dict(api_keys)}
            if config:
                client_kwargs["max_retries"] = config.max_retries_per_key
                client_kwargs["global_timeout"] = config.global_timeout
                client_kwargs["max_concurrent_requests_per_key"] = config.provider_concurrency
            client = RotatingClient(**client_kwargs)

            # Use fixer model from request, or config, or default
            if request.fixer_model:
                fixer_model = request.fixer_model
            elif config:
                fixer_model = config.fixer_model
            else:
                fixer_model = "anthropic/claude-3-5-sonnet-20241022"

            # Get model options for fixer if configured
            fixer_options = config.get_model_options(fixer_model) if config else {}

            code_fixer = CodeFixer(client, fixer_model, fixer_options)

        # Get original response
        original_response = results_manager.get_response(run_id, model_name, question_id)
        if not original_response:
            raise HTTPException(status_code=404, detail="Original response not found")

        # Get question
        question = question_loader.get_question(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Fix the response
        fixed_response = await code_fixer.fix_response(question, original_response)

        # Save the fixed response
        results_manager.save_fixed_response_for_run(run_id, fixed_response)

        return {
            "success": True,
            "message": "Response formatting fixed successfully",
            "fixed_response": fixed_response.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fix failed: {str(e)}")


@app.get("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/versions")
async def list_response_versions(run_id: str, model_name: str, question_id: str):
    """List all versions of a response for a specific question."""
    try:
        versions = results_manager.list_response_versions(run_id, model_name, question_id)
        return {"versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/regenerate")
async def regenerate_response(run_id: str, model_name: str, question_id: str, confirm_replace: bool = False):
    """Regenerate response and evaluation for a question using new regeneration service."""
    try:
        from src.services.regeneration import regenerate_response as regenerate_service
        from src.rotator_client import RotatingClient
        from src.evaluators import LLMJudgeEvaluator, CodeExecutor
        from src.evaluators.tool_validator import ToolCallValidator

        # Get question
        question = question_loader.get_question(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Get run metadata
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Check current instance status
        current_instance_id = results_manager.get_current_instance(run_id, model_name, question_id)
        if current_instance_id:
            current_response = results_manager.get_response(run_id, model_name, question_id)
            has_error = current_response.error is not None if current_response else False

            # If no error and user didn't confirm, require confirmation
            if not has_error and not confirm_replace:
                return {
                    "success": False,
                    "requires_confirmation": True,
                    "message": "Current response is successful. Set confirm_replace=true to proceed with regeneration."
                }

        # Collect API keys
        api_keys = defaultdict(list)
        for key, value in os.environ.items():
            if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
                parts = key.split("_API_KEY")
                provider = parts[0].lower()
                if provider not in api_keys:
                    api_keys[provider] = []
                api_keys[provider].append(value)

        if not api_keys:
            raise HTTPException(status_code=500, detail="No API keys configured")

        # Create client with retry settings from config if available
        client_kwargs = {"api_keys": dict(api_keys)}
        if config:
            client_kwargs["max_retries"] = config.max_retries_per_key
            client_kwargs["global_timeout"] = config.global_timeout
            client_kwargs["max_concurrent_requests_per_key"] = config.provider_concurrency
        client = RotatingClient(**client_kwargs)

        # Initialize evaluators
        judge_model = run.judge_model or "anthropic/claude-3-5-sonnet-20241022"
        evaluators = {
            'llm_judge': LLMJudgeEvaluator(client, judge_model),
            'code_executor': CodeExecutor(),
            'tool_validator': ToolCallValidator()
        }

        # Determine whether to replace current instance
        # If there's an error, always replace; otherwise, respect user's choice
        should_replace = has_error or confirm_replace

        # Call regeneration service
        new_response, evaluation = await regenerate_service(
            run_id=run_id,
            model_name=model_name,
            question_id=question_id,
            question=question,
            client=client,
            results_manager=results_manager,
            evaluators=evaluators,
            replace_current=should_replace,
            model_options=config.get_model_options(model_name) if config else None
        )

        return {
            "success": True,
            "message": "Response regenerated and evaluated successfully",
            "response": new_response.model_dump(),
            "evaluation": evaluation.model_dump() if evaluation else None
        }

    except ValueError as e:
        # Max instances reached or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")


# ============================================================================
# Instance Management API Endpoints
# ============================================================================

@app.get("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/instances")
async def list_instances(run_id: str, model_name: str, question_id: str):
    """List all instances for a specific question."""
    try:
        instances = results_manager.list_instances(run_id, model_name, question_id)
        return {"instances": instances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/instances/{instance_id}")
async def get_instance(run_id: str, model_name: str, question_id: str, instance_id: str):
    """Get a specific instance by ID."""
    try:
        response = results_manager.get_response(run_id, model_name, question_id, instance_id=instance_id)
        if not response:
            raise HTTPException(status_code=404, detail="Instance not found")

        # Get evaluations for this instance
        evaluations = results_manager.get_instance_evaluations(run_id, model_name, question_id, instance_id)

        # Check if this is the current instance
        current_instance_id = results_manager.get_current_instance(run_id, model_name, question_id)
        is_current = (instance_id == current_instance_id)

        return {
            "instance": response.model_dump(),
            "evaluations": evaluations,
            "is_current": is_current
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SetCurrentInstanceRequest(BaseModel):
    """Request body for setting current instance."""
    instance_id: str


@app.put("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/current")
async def set_current_instance(run_id: str, model_name: str, question_id: str, request: SetCurrentInstanceRequest):
    """Set the current instance for a question."""
    try:
        # Verify instance exists
        response = results_manager.get_response(run_id, model_name, question_id, instance_id=request.instance_id)
        if not response:
            raise HTTPException(status_code=404, detail="Instance not found")

        # Set as current
        results_manager.set_current_instance(run_id, model_name, question_id, request.instance_id)

        # Trigger leaderboard recalculation
        # Load the run to get categories and questions
        run = results_manager.get_run(run_id)
        if run:
            # Get all questions for the categories in this run
            questions = [q for q in question_loader.questions.values() if q.category in run.categories]
            results_manager.current_run_dir = results_manager.results_dir / run_id
            results_manager.current_run = run
            results_manager.calculate_and_save_scores(questions)

        return {
            "success": True,
            "message": f"Instance {request.instance_id} set as current",
            "instance_id": request.instance_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/runs/{run_id}/models/{model_name:path}/questions/{question_id}/instances/{instance_id}")
async def delete_instance(run_id: str, model_name: str, question_id: str, instance_id: str):
    """Delete a specific instance (cannot delete current instance)."""
    try:
        results_manager.delete_instance(run_id, model_name, question_id, instance_id)

        return {
            "success": True,
            "message": f"Instance {instance_id} deleted successfully"
        }
    except ValueError as e:
        # Current instance or other validation error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/questions/{question_id}")
async def get_question_responses(run_id: str, question_id: str):
    """Get model response for a specific question in a run."""
    try:
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        question = question_loader.get_question(question_id)

        # Each run now has only one model
        response = results_manager.get_response(run_id, run.model, question_id)
        evaluation = results_manager.get_evaluation(run_id, run.model, question_id)

        return {
            "question": question.model_dump() if question else None,
            "response": response.model_dump() if response else None,
            "evaluation": evaluation.model_dump() if evaluation else None,
            "model": run.model
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/questions")
async def get_questions():
    """Get all available questions."""
    try:
        questions = list(question_loader.questions.values())
        return {
            "questions": [q.model_dump() for q in questions]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    """Get all available categories."""
    try:
        categories = question_loader.get_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories/detailed")
async def get_categories_detailed():
    """Get categories with metadata (question counts, display names)."""
    try:
        categories = question_loader.get_categories()
        category_data = []

        for cat in categories:
            # Load questions for this category
            questions = question_loader.load_category(cat)

            # Create display name (convert snake_case to Title Case)
            display_name = cat.replace('_', ' ').title()

            category_data.append({
                "name": cat,
                "display_name": display_name,
                "question_count": len(questions)
            })

        return {"categories": category_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-display-names")
async def get_model_display_names():
    """Get friendly display names for models from config."""
    try:
        if config:
            display_names = config.model_display_names
        else:
            display_names = {}
        return {"model_display_names": display_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name:path}/runs")
async def get_model_runs(model_name: str):
    """Get all benchmark runs containing a specific model."""
    try:
        runs = results_manager.get_model_runs(model_name)
        return {"model_name": model_name, "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leaderboard/unified")
async def get_unified_leaderboard():
    """Get unified leaderboard using preferred runs per model."""
    try:
        leaderboard = results_manager.get_unified_leaderboard()
        return {"leaderboard": leaderboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leaderboard/all-runs")
async def get_all_runs_leaderboard():
    """Get expanded leaderboard showing all runs as separate entries."""
    try:
        leaderboard = results_manager.get_all_runs_leaderboard()
        return {"leaderboard": leaderboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leaderboard/preferences")
async def get_leaderboard_preferences():
    """Get leaderboard preferences."""
    try:
        preferences = results_manager.get_leaderboard_preferences()
        return {"preferences": preferences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/leaderboard/preferences/{model_name}")
async def set_leaderboard_preference(model_name: str, request: PreferenceRequest):
    """Set leaderboard preference for a model."""
    try:
        results_manager.set_leaderboard_preference(model_name, request.run_id)
        return {"success": True, "model_name": model_name, "run_id": request.run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/leaderboard/preferences/{model_name}")
async def clear_leaderboard_preference(model_name: str):
    """Clear leaderboard preference for a model."""
    try:
        results_manager.clear_leaderboard_preference(model_name)
        return {"success": True, "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExecuteRequest(BaseModel):
    """Request body for executing code."""
    code: str
    language: str = "python"
    timeout: int = 10
    args: str = ""  # Command-line arguments
    stdin: str = ""  # Standard input data
    session_id: str = ""  # Session identifier for persistent temp directories (format: runId_modelName_questionId_version)


@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    """Execute code and return output."""
    import subprocess
    import tempfile
    import os
    import shlex

    try:
        # Use persistent temp directory if session_id provided, otherwise create new one
        if request.session_id and request.session_id in persistent_temp_dirs:
            temp_dir = persistent_temp_dirs[request.session_id]
            cleanup_temp = False  # Don't cleanup persistent directories
        elif request.session_id:
            # Create new persistent temp directory for this session
            temp_dir = tempfile.mkdtemp(prefix=f'viewer_exec_{request.session_id.replace("/", "_")}_')
            persistent_temp_dirs[request.session_id] = temp_dir
            cleanup_temp = False  # Don't cleanup persistent directories
        else:
            # No session ID - create temporary directory (old behavior)
            temp_dir = tempfile.mkdtemp(prefix='viewer_exec_')
            cleanup_temp = True  # Cleanup non-persistent directories

        try:
            # Parse command-line arguments
            args_list = []
            if request.args:
                try:
                    args_list = shlex.split(request.args)
                except ValueError:
                    # If shlex fails, split by spaces
                    args_list = request.args.split()

            if request.language == "python":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'script.py')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Build command with arguments
                cmd = ['python', temp_file] + args_list

                # Execute
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "javascript":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'script.js')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Build command with arguments
                cmd = ['node', temp_file] + args_list

                # Execute with node
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "rust":
                # Create Cargo project structure
                project_dir = os.path.join(temp_dir, 'rust_project')
                os.makedirs(project_dir)
                src_dir = os.path.join(project_dir, 'src')
                os.makedirs(src_dir)

                # Write main.rs
                main_file = os.path.join(src_dir, 'main.rs')
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Write Cargo.toml
                cargo_toml = os.path.join(project_dir, 'Cargo.toml')
                with open(cargo_toml, 'w', encoding='utf-8') as f:
                    f.write("""[package]
name = "temp_rust"
version = "0.1.0"
edition = "2021"

[dependencies]
""")

                # Build command
                cmd = ['cargo', 'run', '--quiet'] + (['--'] + args_list if args_list else [])

                # Execute
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=project_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "go":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'main.go')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Build command
                cmd = ['go', 'run', 'main.go'] + args_list

                # Execute
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language in ["cpp", "c++"]:
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'main.cpp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Compile
                output_file = os.path.join(temp_dir, 'program.exe' if os.name == 'nt' else 'program')
                compile_result = subprocess.run(
                    ['g++', '-std=c++17', temp_file, '-o', output_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Compilation failed:\n{compile_result.stderr}",
                        "returncode": compile_result.returncode
                    }

                # Execute
                cmd = [output_file] + args_list
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "c":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'main.c')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Compile
                output_file = os.path.join(temp_dir, 'program.exe' if os.name == 'nt' else 'program')
                compile_result = subprocess.run(
                    ['gcc', temp_file, '-o', output_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Compilation failed:\n{compile_result.stderr}",
                        "returncode": compile_result.returncode
                    }

                # Execute
                cmd = [output_file] + args_list
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "ruby":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'script.rb')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Build command with arguments
                cmd = ['ruby', temp_file] + args_list

                # Execute
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "php":
                # Write code to temp file
                temp_file = os.path.join(temp_dir, 'script.php')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Build command with arguments
                cmd = ['php', temp_file] + args_list

                # Execute
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif request.language == "java":
                # Extract class name from code
                import re
                class_match = re.search(r'public\s+class\s+(\w+)', request.code)
                class_name = class_match.group(1) if class_match else 'Main'

                # Write code to temp file
                temp_file = os.path.join(temp_dir, f'{class_name}.java')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(request.code)

                # Compile
                compile_result = subprocess.run(
                    ['javac', f'{class_name}.java'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Compilation failed:\n{compile_result.stderr}",
                        "returncode": compile_result.returncode
                    }

                # Execute
                cmd = ['java', class_name] + args_list
                result = subprocess.run(
                    cmd,
                    input=request.stdin if request.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout,
                    encoding='utf-8',
                    errors='replace',
                    cwd=temp_dir
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        finally:
            # Cleanup only if not a persistent directory
            if cleanup_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {request.timeout} seconds",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


# ============================================================================
# Comparative Judge API Endpoints
# ============================================================================

from src.job_runner import get_job_runner
from src.evaluators.comparative_judge import ComparativeJudgeEvaluator


class ComparativeJudgeRequest(BaseModel):
    """Request body for starting comparative judge job."""
    run_ids: List[str]
    question_ids: Optional[List[str]] = None  # If None, use all questions
    mode: str = "normal"  # "normal" or "show_all"


@app.post("/api/comparative-judge/start")
async def start_comparative_judge(request: ComparativeJudgeRequest):
    """Start a new comparative judge job."""
    try:
        # Get job runner
        job_runner = get_job_runner(results_manager.results_dir)

        # If no question IDs specified, use all questions
        question_ids = request.question_ids
        if not question_ids:
            question_ids = list(question_loader.questions.keys())

        # Filter run_ids based on mode
        run_ids_to_compare = request.run_ids

        if request.mode == "normal":
            # Normal mode: Use preferred runs only (one run per model)
            # Group runs by model and select preferred run per model
            model_to_runs = {}
            for run_id in request.run_ids:
                run = results_manager.get_run(run_id)
                if run:
                    model_name = run.model
                    if model_name not in model_to_runs:
                        model_to_runs[model_name] = []
                    model_to_runs[model_name].append(run_id)

            # Get preferences
            preferences = results_manager.get_leaderboard_preferences()

            # Select one run per model (preferred or latest)
            run_ids_to_compare = []
            for model_name, run_list in model_to_runs.items():
                if model_name in preferences:
                    # Use preferred run if it's in the list
                    if preferences[model_name] in run_list:
                        run_ids_to_compare.append(preferences[model_name])
                    else:
                        # Fallback to latest
                        run_ids_to_compare.append(run_list[-1])
                else:
                    # Use latest run
                    run_ids_to_compare.append(run_list[-1])
        # else: show_all mode uses all run_ids as-is

        # Create job
        job_id = job_runner.create_job(run_ids_to_compare, question_ids)

        # Initialize evaluator
        from collections import defaultdict
        api_keys = defaultdict(list)
        for key, value in os.environ.items():
            if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
                parts = key.split("_API_KEY")
                provider = parts[0].lower()
                if provider not in api_keys:
                    api_keys[provider] = []
                api_keys[provider].append(value)

        if not api_keys:
            raise HTTPException(status_code=500, detail="No API keys configured")

        from src.rotator_client import RotatingClient
        client_kwargs = {"api_keys": dict(api_keys)}
        if config:
            client_kwargs["max_retries"] = config.max_retries_per_key
            client_kwargs["global_timeout"] = config.global_timeout
            client_kwargs["max_concurrent_requests_per_key"] = config.provider_concurrency
        client = RotatingClient(**client_kwargs)

        judge_model = config.judge_model if config else "anthropic/claude-3-5-sonnet-20241022"
        evaluator = ComparativeJudgeEvaluator(client, judge_model)

        # Get concurrency settings from config
        max_concurrent = config.max_concurrent if config else 3
        provider_concurrency = config.provider_concurrency if config else None

        # Start job in background with concurrency settings
        job_runner.start_job(
            job_id,
            evaluator,
            question_loader,
            results_manager,
            max_concurrent,
            provider_concurrency
        )

        return {"job_id": job_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comparative-judge/jobs/{job_id}/status")
async def get_comparative_judge_status(job_id: str):
    """Get status of a comparative judge job."""
    try:
        job_runner = get_job_runner(results_manager.results_dir)
        job = job_runner.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return job.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comparative-judge/jobs/{job_id}/results")
async def get_comparative_judge_results(job_id: str):
    """Get results of a completed comparative judge job."""
    try:
        job_runner = get_job_runner(results_manager.results_dir)
        results = job_runner.load_job_results(job_id)

        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comparative-judge/jobs")
async def get_all_comparative_judge_jobs():
    """Get all comparative judge jobs."""
    try:
        job_runner = get_job_runner(results_manager.results_dir)
        jobs = job_runner.get_all_jobs()

        return {"jobs": [job.to_dict() for job in jobs]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/comparative-judge/jobs/{job_id}")
async def cancel_comparative_judge_job(job_id: str):
    """Cancel a running comparative judge job."""
    try:
        job_runner = get_job_runner(results_manager.results_dir)
        cancelled = job_runner.cancel_job(job_id)

        if not cancelled:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled (not found or not running)")

        return {"success": True, "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Benchmark Runner API Endpoints
# ============================================================================

from viewer.async_benchmark_runner import get_async_runner
from viewer.benchmark_job_manager import get_job_manager


class BenchmarkRequest(BaseModel):
    """Request body for starting a benchmark run."""
    models: List[str]
    categories: Optional[List[str]] = None
    question_ids: Optional[List[str]] = None
    max_concurrent: int = 10
    provider_concurrency: Optional[Dict[str, int]] = None


@app.post("/api/benchmark/start")
async def start_benchmark(request: BenchmarkRequest):
    """Start a new benchmark run."""
    try:
        job_manager = get_job_manager()

        # Check if already running
        if job_manager.is_running():
            raise HTTPException(status_code=400, detail="A benchmark is already running")

        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"benchmark_{timestamp}"

        # Load full config
        if not config:
            raise HTTPException(status_code=500, detail="Configuration not loaded")

        # Build config dict for benchmark
        benchmark_config = {
            "models": request.models,
            "categories": request.categories,
            "question_ids": request.question_ids,
            "max_concurrent": request.max_concurrent,
            "provider_concurrency": request.provider_concurrency,
            "judge_model": config.judge_model,
            "questions_dir": config.questions_dir,
            "results_dir": config.results_dir,
            "model_configs": config.model_configs,
            "code_formatting_instructions": {
                "enabled": config.code_formatting_enabled,
                "instruction": config.code_formatting_instruction
            }
        }

        # Get API keys
        from collections import defaultdict
        api_keys = defaultdict(list)
        for key, value in os.environ.items():
            if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
                parts = key.split("_API_KEY")
                provider = parts[0].lower()
                if provider not in api_keys:
                    api_keys[provider] = []
                api_keys[provider].append(value)

        if not api_keys:
            raise HTTPException(status_code=500, detail="No API keys configured")

        # Create RotatingClient
        from src.rotator_client import RotatingClient
        client_kwargs = {"api_keys": dict(api_keys)}
        client_kwargs["max_retries"] = config.max_retries_per_key
        client_kwargs["global_timeout"] = config.global_timeout
        client_kwargs["max_concurrent_requests_per_key"] = config.provider_concurrency
        client = RotatingClient(**client_kwargs)

        # Start benchmark in background
        runner = get_async_runner()
        await runner.start_benchmark(job_id, client, benchmark_config)

        return {"job_id": job_id, "status": "started"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark/status")
async def get_benchmark_status():
    """Get status of the current benchmark run."""
    try:
        job_manager = get_job_manager()
        job = job_manager.get_current_job()

        if not job:
            return {"status": "idle", "job": None}

        return {"status": "running" if job_manager.is_running() else job["status"], "job": job}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/benchmark/stop")
async def stop_benchmark():
    """Stop the current benchmark run."""
    try:
        runner = get_async_runner()
        runner.cancel()

        return {"success": True, "message": "Benchmark cancellation requested"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark/history")
async def get_benchmark_history(limit: int = 10):
    """Get benchmark job history."""
    try:
        job_manager = get_job_manager()
        history = job_manager.get_history(limit=limit)

        return {"history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark/jobs/{job_id}")
async def get_benchmark_job(job_id: str):
    """Get details of a specific benchmark job."""
    try:
        job_manager = get_job_manager()
        job = job_manager.get_job_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return job

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Human Ratings API Endpoints
# ============================================================================

class HumanRatingRequest(BaseModel):
    """Request body for saving human rating."""
    score: float  # 0-100, supports decimals
    comment: Optional[str] = None


@app.get("/api/runs/{run_id}/human-ratings/{model_name:path}/{question_id}")
async def get_human_rating(run_id: str, model_name: str, question_id: str):
    """Get human rating for a specific response."""
    try:
        rating_file = results_manager.results_dir / run_id / "human_ratings" / results_manager._sanitize_name(model_name) / f"{question_id}.json"

        if not rating_file.exists():
            return {"rating": None}

        with open(rating_file, 'r') as f:
            rating = json.load(f)

        return {"rating": rating}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/{run_id}/human-ratings/{model_name:path}/{question_id}")
async def save_human_rating(run_id: str, model_name: str, question_id: str, request: HumanRatingRequest):
    """Save human rating for a specific response."""
    try:
        # Validate score range
        if not 0 <= request.score <= 100:
            raise HTTPException(status_code=400, detail="Score must be between 0 and 100")

        # Create directory structure
        ratings_dir = results_manager.results_dir / run_id / "human_ratings" / results_manager._sanitize_name(model_name)
        ratings_dir.mkdir(parents=True, exist_ok=True)

        # Save rating
        rating_file = ratings_dir / f"{question_id}.json"
        rating_data = {
            "score": request.score,
            "comment": request.comment,
            "timestamp": datetime.now().isoformat()
        }

        with open(rating_file, 'w') as f:
            json.dump(rating_data, f, indent=2)

        return {"success": True, "rating": rating_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/human-ratings/leaderboard")
async def get_human_ratings_leaderboard(run_id: str):
    """Get leaderboard based on human ratings for a run."""
    try:
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        ratings_dir = results_manager.results_dir / run_id / "human_ratings"
        if not ratings_dir.exists():
            return {"leaderboard": []}

        # Collect ratings per model
        model_ratings = {}

        for model_dir in ratings_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            ratings = []

            for rating_file in model_dir.glob("*.json"):
                with open(rating_file, 'r') as f:
                    rating_data = json.load(f)
                    ratings.append(rating_data['score'])

            if ratings:
                model_ratings[model_name] = {
                    'average_score': sum(ratings) / len(ratings),
                    'total_rated': len(ratings),
                    'min_score': min(ratings),
                    'max_score': max(ratings)
                }

        # Build leaderboard
        leaderboard = [
            {
                'model_name': model_name,
                **data
            }
            for model_name, data in model_ratings.items()
        ]

        # Sort by average score
        leaderboard.sort(key=lambda x: x['average_score'], reverse=True)

        return {"leaderboard": leaderboard}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/human-ratings/unified-leaderboard")
async def get_unified_human_ratings_leaderboard():
    """Get unified leaderboard based on human ratings across all runs."""
    try:
        # Collect ratings per model across ALL runs
        model_ratings = {}

        # Iterate through all runs
        for run_dir in results_manager.results_dir.iterdir():
            if not run_dir.is_dir():
                continue

            ratings_dir = run_dir / "human_ratings"
            if not ratings_dir.exists():
                continue

            # Process each model's ratings in this run
            for model_dir in ratings_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name

                # Initialize model entry if not exists
                if model_name not in model_ratings:
                    model_ratings[model_name] = {
                        'ratings': [],
                        'runs': set()
                    }

                # Collect all ratings for this model in this run
                for rating_file in model_dir.glob("*.json"):
                    try:
                        with open(rating_file, 'r') as f:
                            rating_data = json.load(f)
                            model_ratings[model_name]['ratings'].append(rating_data['score'])
                            model_ratings[model_name]['runs'].add(run_dir.name)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # Build unified leaderboard
        leaderboard = []
        for model_name, data in model_ratings.items():
            ratings = data['ratings']
            if ratings:
                leaderboard.append({
                    'model_name': model_name,
                    'average_score': sum(ratings) / len(ratings),
                    'total_rated': len(ratings),
                    'min_score': min(ratings),
                    'max_score': max(ratings),
                    'runs_count': len(data['runs'])
                })

        # Sort by average score (descending)
        leaderboard.sort(key=lambda x: x['average_score'], reverse=True)

        return {"leaderboard": leaderboard}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Author's Choice API Endpoints
# ============================================================================

class AuthorsChoiceRequest(BaseModel):
    """Request body for saving author's choice rankings."""
    rankings: List[Dict[str, Any]]  # [{model_name, position}, ...]


@app.get("/api/authors-choice")
async def get_authors_choice():
    """Get author's choice rankings."""
    try:
        user_data_dir = Path("user_data")
        ranking_file = user_data_dir / "authors_choice.json"

        if not ranking_file.exists():
            return {"rankings": []}

        with open(ranking_file, 'r') as f:
            data = json.load(f)

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/authors-choice")
async def save_authors_choice(request: AuthorsChoiceRequest):
    """Save author's choice rankings."""
    try:
        user_data_dir = Path("user_data")
        user_data_dir.mkdir(exist_ok=True)

        ranking_file = user_data_dir / "authors_choice.json"

        data = {
            "rankings": request.rankings,
            "last_updated": datetime.now().isoformat()
        }

        with open(ranking_file, 'w') as f:
            json.dump(data, f, indent=2)

        return {"success": True, "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Config Editor API Endpoints
# ============================================================================

class ConfigValidateRequest(BaseModel):
    """Request body for validating config."""
    yaml_content: str


class ConfigSaveRequest(BaseModel):
    """Request body for saving config."""
    yaml_content: str


@app.get("/api/config")
async def get_config():
    """Get current config.yaml content."""
    try:
        # Look for config.yaml in parent directory (project root)
        config_path = Path(__file__).parent.parent / "config.yaml"

        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"config.yaml not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {"content": content}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/validate")
async def validate_config(request: ConfigValidateRequest):
    """Validate YAML config without saving."""
    try:
        import yaml

        # Try to parse YAML
        try:
            parsed = yaml.safe_load(request.yaml_content)
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "errors": [f"YAML syntax error: {str(e)}"]
            }

        # Basic structure validation
        errors = []

        if not isinstance(parsed, dict):
            errors.append("Config must be a YAML object/dictionary")

        # Check for required top-level keys
        if isinstance(parsed, dict):
            if 'models' not in parsed:
                errors.append("Missing required key: 'models'")
            elif not isinstance(parsed['models'], list):
                errors.append("'models' must be a list")

        if errors:
            return {"valid": False, "errors": errors}

        return {"valid": True, "errors": []}

    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"]
        }


@app.post("/api/config/save")
async def save_config(request: ConfigSaveRequest):
    """Save config.yaml with automatic backup."""
    try:
        import yaml

        # Use parent directory (project root)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"

        # Validate first
        try:
            yaml.safe_load(request.yaml_content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

        # Create backup with sequential numbering (bak1, bak2, bak3, etc.)
        if config_path.exists():
            # Find the next available backup number
            backup_num = 1
            while True:
                backup_path = project_root / f"config.yaml.bak{backup_num}"
                if not backup_path.exists():
                    break
                backup_num += 1

            shutil.copy(config_path, backup_path)
            backup_created = str(backup_path)
        else:
            backup_created = None

        # Save new config
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(request.yaml_content)

        return {
            "success": True,
            "backup_path": backup_created,
            "message": "Config saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/backups")
async def get_config_backups():
    """Get list of config backups."""
    try:
        # Look in parent directory (project root)
        project_root = Path(__file__).parent.parent
        backup_files = list(project_root.glob("config.yaml.bak*"))

        backups = []
        for backup_file in backup_files:
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        backups.sort(key=lambda x: x['modified'], reverse=True)

        return {"backups": backups}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/restore/{backup_name}")
async def restore_config(backup_name: str):
    """Restore config from backup."""
    try:
        # Use parent directory (project root)
        project_root = Path(__file__).parent.parent
        backup_path = project_root / backup_name

        if not backup_path.exists() or not backup_path.name.startswith("config.yaml.bak"):
            raise HTTPException(status_code=404, detail="Backup not found")

        config_path = project_root / "config.yaml"

        # Create backup of current config before restoring
        if config_path.exists():
            temp_backup = project_root / "config.yaml.bak.temp"
            shutil.copy(config_path, temp_backup)

        # Restore from backup
        shutil.copy(backup_path, config_path)

        return {
            "success": True,
            "message": f"Config restored from {backup_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/config/backups/{backup_name}")
async def delete_config_backup(backup_name: str):
    """Delete a specific config backup."""
    try:
        # Use parent directory (project root)
        project_root = Path(__file__).parent.parent
        backup_path = project_root / backup_name

        # Validate backup file name and existence
        if not backup_path.exists() or not backup_path.name.startswith("config.yaml.bak"):
            raise HTTPException(status_code=404, detail="Backup not found")

        # Delete the backup file
        backup_path.unlink()

        return {
            "success": True,
            "message": f"Backup {backup_name} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount artifacts directory for multi-file web apps
artifacts_dir = artifact_extractor.temp_base_dir
artifacts_dir.mkdir(exist_ok=True, parents=True)
app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")


def main():
    """Start the server."""
    print("\n" + "="*60)
    print("MirroBench - LLM Benchmark Viewer")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8080")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
