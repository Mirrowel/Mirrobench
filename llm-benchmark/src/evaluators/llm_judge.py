"""
LLM-as-judge evaluator using a powerful model to evaluate responses.
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from lib.rotator_library.client import RotatingClient
from src.schemas import Question, ModelResponse, Evaluation


class LLMJudgeEvaluator:
    """Use an LLM to evaluate model responses."""

    def __init__(
        self,
        client: RotatingClient,
        judge_model: str = "anthropic/claude-3-5-sonnet-20241022",
        model_options: Optional[Dict[str, Any]] = None
    ):
        self.client = client
        self.judge_model = judge_model
        self.model_options = model_options or {}

    async def evaluate(
        self,
        question: Question,
        response: ModelResponse,
        code_execution_result: Optional[Evaluation] = None
    ) -> Evaluation:
        """
        Evaluate a model response using an LLM judge.

        Args:
            question: The original question
            response: The model's response to evaluate
            code_execution_result: Optional code execution evaluation result to inform the judge

        Returns:
            Evaluation: The evaluation result
        """
        # If there was an error in the response, automatically fail
        if response.error:
            return Evaluation(
                question_id=question.id,
                model_name=response.model_name,                score=0.0,
                passed=False,
                evaluation_type="llm_judge",
                evaluator_model=self.judge_model,
                reasoning=f"Response failed with error: {response.error}",
                details={"error": response.error},
                timestamp=datetime.now().isoformat()
            )

        # Build the evaluation prompt
        eval_prompt = self._build_evaluation_prompt(question, response, code_execution_result)

        try:
            # Build request kwargs
            kwargs = {
                "model": self.judge_model,
                "messages": [
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ],
                "temperature": 0.0,  # Deterministic evaluation
                "stream": False  # Important: disable streaming for evaluation
            }

            # Add model-specific options if configured for judge model
            if self.model_options:
                kwargs.update(self.model_options)

            # Call the judge model (non-streaming for evaluation)
            judge_response = await self.client.acompletion(**kwargs)

            # Extract the response text
            judge_text = None
            if judge_response and hasattr(judge_response, 'choices'):
                if len(judge_response.choices) > 0:
                    message = judge_response.choices[0].message

                    # Try content first
                    if hasattr(message, 'content') and message.content:
                        judge_text = message.content
                    # For extended thinking models (e.g., Gemini 2.5), try reasoning_content
                    elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                        judge_text = message.reasoning_content

            # Fallback: try to parse as dict
            if judge_text is None and isinstance(judge_response, dict):
                if 'choices' in judge_response and len(judge_response['choices']) > 0:
                    msg = judge_response['choices'][0].get('message', {})
                    judge_text = msg.get('content') or msg.get('reasoning_content')

            # Last resort: convert to string (but avoid the repr of ModelResponse)
            if judge_text is None:
                if judge_response and not isinstance(judge_response, str):
                    # Don't use str() on complex objects - it creates ugly output
                    raise ValueError("Could not extract text from judge response")
                judge_text = str(judge_response) if judge_response else ""

            # Validate we got text
            if not judge_text or judge_text == "None":
                raise ValueError("Judge model returned empty response")

            # Parse the judge's response
            score, passed, reasoning = self._parse_judge_response(judge_text)

            # Check if response was truncated
            was_truncated = False
            if hasattr(judge_response, 'choices') and len(judge_response.choices) > 0:
                finish_reason = getattr(judge_response.choices[0], 'finish_reason', None)
                if finish_reason == 'length':
                    was_truncated = True
                    reasoning = f"[TRUNCATED] {reasoning}"

            return Evaluation(
                question_id=question.id,
                model_name=response.model_name,
                score=score,
                passed=passed,
                evaluation_type="llm_judge",
                evaluator_model=self.judge_model,
                reasoning=reasoning,
                details={
                    "full_judge_response": judge_text,
                    "was_truncated": was_truncated
                },
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            # If evaluation fails, return a default failure
            return Evaluation(
                question_id=question.id,
                model_name=response.model_name,                score=0.0,
                passed=False,
                evaluation_type="llm_judge",
                evaluator_model=self.judge_model,
                reasoning=f"Evaluation failed: {str(e)}",
                details={"evaluation_error": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _build_evaluation_prompt(
        self,
        question: Question,
        response: ModelResponse,
        code_execution_result: Optional[Evaluation] = None
    ) -> str:
        """Build the prompt for the judge model."""
        criteria = question.evaluation_criteria or "Evaluate the response for correctness, completeness, and quality."

        # Don't truncate response - judge needs to see everything
        prompt = f"""You are an expert evaluator assessing an AI model's response to a question.

**Question:**
{question.prompt}

**Model's Response:**
{response.response_text}

**Evaluation Criteria:**
{criteria}
"""

        if question.expected_output:
            prompt += f"""
**Expected Output (for reference):**
{question.expected_output}
"""

        # Add code execution results if provided
        if code_execution_result:
            prompt += f"""
**Code Execution Results:**
- Status: {'PASSED' if code_execution_result.passed else 'FAILED'}
- Score: {code_execution_result.score}/100
- Reasoning: {code_execution_result.reasoning}
"""
            if code_execution_result.details:
                if 'error' in code_execution_result.details:
                    prompt += f"""- Error: {code_execution_result.details['error']}
"""
                if 'output' in code_execution_result.details:
                    prompt += f"""- Output: {code_execution_result.details['output']}
"""

            prompt += """
**IMPORTANT:** The code execution test has already run. Consider this technical validation in your evaluation. If the code failed to execute properly, this should heavily influence your score, even if the code looks reasonable at first glance.
"""

        prompt += """

**Your Task:**
Evaluate the response and provide:
1. A score from 0-100 (where 0 is completely wrong and 100 is perfect and beautiful)
2. Whether the response passes (score >= 70)
3. Brief reasoning for your score

**Scoring Guidelines - BE HARSH BUT FAIR:**
- **100**: ONLY for responses that are PERFECT, BEAUTIFUL, and go above and beyond expectations. Flawless execution, excellent code quality, comprehensive documentation, exceptional user experience.
- **90-99**: Excellent responses with minor imperfections or missing edge cases.
- **80-89**: Good responses that work correctly but lack polish, optimization, or best practices.
- **70-79**: Adequate responses that meet basic requirements but have notable issues or inefficiencies.
- **60-69**: Responses that partially work but have significant issues or incomplete functionality.
- **40-59**: Poor responses with major errors or missing key functionality.
- **20-39**: Severely flawed responses with fundamental misunderstandings.
- **0-19**: Completely wrong or non-functional responses.

**Evaluation Standards:**
- Scrutinize code quality, error handling, edge cases, and best practices
- Expect proper documentation and clear structure
- Demand efficiency and optimization
- Look for security vulnerabilities or bad practices
- Consider user experience and completeness
- Be critical of shortcuts, hacks, or incomplete solutions
- A "working" solution is NOT automatically a high score - quality matters

**CRITICAL: You MUST format your final assessment EXACTLY as follows:**
SCORE: [number from 0-100]
PASSED: [YES or NO]
REASONING: [Your explanation]

**Note:** You can think through your evaluation first, but you MUST end with the exact SCORE/PASSED/REASONING format above.

Be objective, critical, and fair in your evaluation. Reserve high scores for truly excellent work."""

        return prompt

    def _parse_judge_response(self, judge_text: str) -> tuple:
        """
        Parse the judge's response to extract score, pass/fail, and reasoning.
        Handles both structured format and thinking/reasoning content.

        Returns:
            tuple: (score, passed, reasoning)
        """
        try:
            import re

            # Look for structured format anywhere in the text (not just at start)
            # This handles thinking models that put reasoning first
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', judge_text, re.IGNORECASE)
            passed_match = re.search(r'PASSED:\s*(YES|NO|TRUE|FALSE|PASS|FAIL)', judge_text, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', judge_text, re.IGNORECASE | re.DOTALL)

            score = 0.0
            passed = False
            reasoning = ""

            if score_match:
                score = float(score_match.group(1))

            if passed_match:
                passed_str = passed_match.group(1).upper()
                passed = passed_str in ["YES", "TRUE", "PASS"]

            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Don't truncate - keep full reasoning

            # Ensure score is in valid range
            score = max(0.0, min(100.0, score))

            # Fallback: if parsing failed but we can infer from text
            if score == 0.0:
                text_lower = judge_text.lower()
                if "excellent" in text_lower or "perfect" in text_lower:
                    score = 90.0
                    passed = True
                elif "good" in text_lower or "well" in text_lower:
                    score = 75.0
                    passed = True
                elif "acceptable" in text_lower or "adequate" in text_lower:
                    score = 60.0
                    passed = False
                elif "poor" in text_lower or "fail" in text_lower or "incorrect" in text_lower:
                    score = 30.0
                    passed = False
                else:
                    # If we have thinking/reasoning but no clear format, give moderate score
                    score = 50.0
                    passed = False

            # Set passed based on score if not already set
            if score >= 70:
                passed = True

            if not reasoning:
                # Use full judge text as reasoning
                reasoning = judge_text

            return score, passed, reasoning

        except Exception as e:
            # Fallback on parsing error
            return 50.0, False, f"Failed to parse judge response: {str(e)}"
