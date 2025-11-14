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
from typing import Optional, List, Dict
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


# Load configuration
try:
    config = ConfigLoader()
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

# Initialize managers
results_manager = ResultsManager()
question_loader = QuestionLoader()

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
async def get_response(run_id: str, model_name: str, question_id: str, use_fixed: bool = False, version: Optional[str] = None):
    """Get a specific response with all evaluations, supports version parameter."""
    try:
        # Get the response (original, fixed, or specific version)
        response = results_manager.get_response(run_id, model_name, question_id, use_fixed=use_fixed, version=version)
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
        from lib.rotator_library.client import RotatingClient

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

        # Always run LLM judge
        judge_eval = await llm_judge.evaluate(question, response)
        judge_eval.evaluation_type = "llm_judge"
        results_manager.save_evaluation(judge_eval)
        evaluations_run.append("llm_judge")

        # Run code executor for code execution questions
        if question.evaluation_type == "code_execution":
            code_eval = await code_executor.evaluate(question, response)
            results_manager.save_evaluation(code_eval)
            evaluations_run.append("code_execution")

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
            from lib.rotator_library.client import RotatingClient

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
async def regenerate_response(run_id: str, model_name: str, question_id: str):
    """Regenerate response and evaluation for a question (creates new version)."""
    try:
        from src.runner import BenchmarkRunner
        from lib.rotator_library.client import RotatingClient
        from src.config_loader import ConfigLoader

        # Get question
        question = question_loader.get_question(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Get run metadata
        run = results_manager.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Load config to get model settings
        try:
            config = ConfigLoader("config.yaml")
        except (FileNotFoundError, ValueError):
            # If config not found, use empty defaults
            config = None

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
        client = RotatingClient(**client_kwargs)

        # Initialize runner with run configuration
        judge_model = run.judge_model or "anthropic/claude-3-5-sonnet-20241022"
        runner = BenchmarkRunner(
            client=client,
            judge_model=judge_model,
            results_dir=str(results_manager.results_dir),
            model_system_instructions=config.all_model_system_instructions if config else {},
            model_options=config.all_model_options if config else {},
            code_formatting_enabled=config.code_formatting_enabled if config else True,
            code_formatting_instruction=config.code_formatting_instruction if config else None
        )

        # Set the current run directory
        runner.results_manager.current_run_dir = results_manager.results_dir / run_id

        # Generate new response
        new_response = await runner._generate_response(model_name, question)

        # Save new versioned response
        results_manager.current_run_dir = results_manager.results_dir / run_id
        results_manager.save_response(new_response)  # Will auto-generate new version

        # Evaluate new response
        evaluation = await runner._evaluate_response(question, new_response)

        return {
            "success": True,
            "message": "Response regenerated and evaluated successfully",
            "response": new_response.model_dump(),
            "evaluation": evaluation.model_dump() if evaluation else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")


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
    print("🔴 MirroBench - LLM Benchmark Viewer")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8080")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
