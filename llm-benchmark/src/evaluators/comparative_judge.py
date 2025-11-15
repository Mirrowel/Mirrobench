"""
Comparative Judge Evaluator - Evaluates multiple model responses side-by-side.

This evaluator takes all model responses for a single question and compares them
against each other, providing 0-100 scores for each model in the context of the
other responses. This allows for relative quality assessment.
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.schemas import Question, ModelResponse


class ComparativeJudgeEvaluator:
    """
    Evaluates multiple model responses by comparing them side-by-side.

    Unlike the standard LLM judge which evaluates responses in isolation,
    this evaluator shows the judge all responses at once (anonymized) and
    asks it to score each based on comparative quality.
    """

    def __init__(self, client, judge_model: str):
        """
        Initialize the comparative judge evaluator.

        Args:
            client: RotatingClient instance for API calls
            judge_model: Model to use for judging (e.g., "anthropic/claude-3-5-sonnet-20241022")
        """
        self.client = client
        self.judge_model = judge_model

    async def evaluate_question(
        self,
        question: Question,
        responses: Dict[str, ModelResponse],
        code_execution_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all responses for a single question comparatively.

        Args:
            question: The question being evaluated
            responses: Dict mapping model_name -> ModelResponse
            code_execution_results: Optional dict of code execution results per model

        Returns:
            Dict mapping model_name -> {score, reasoning, details}
        """
        if not responses:
            return {}

        # Anonymize and shuffle responses to avoid bias
        anonymized_responses = self._anonymize_responses(responses)

        # Build the comparative evaluation prompt
        prompt = self._build_comparative_prompt(
            question,
            anonymized_responses,
            code_execution_results
        )

        # Get judge's evaluation
        judge_response = await self._call_judge(prompt)

        # Parse the scores and reasoning
        results = self._parse_comparative_response(
            judge_response,
            anonymized_responses
        )

        return results

    def _anonymize_responses(self, responses: Dict[str, ModelResponse]) -> List[Dict[str, Any]]:
        """
        Anonymize and shuffle responses to prevent model name bias.

        Returns:
            List of dicts with: {label, model_name, response_text, reasoning}
        """
        # Create anonymized entries
        anonymized = []
        for model_name, response in responses.items():
            anonymized.append({
                'label': None,  # Will be assigned after shuffle
                'model_name': model_name,
                'response_text': response.response_text,
                'reasoning': response.reasoning_content
            })

        # Shuffle to remove ordering bias
        random.shuffle(anonymized)

        # Assign labels (Model A, Model B, etc.)
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        for i, entry in enumerate(anonymized):
            entry['label'] = labels[i] if i < len(labels) else f"Model_{i+1}"

        return anonymized

    def _build_comparative_prompt(
        self,
        question: Question,
        anonymized_responses: List[Dict[str, Any]],
        code_execution_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the prompt for comparative evaluation."""

        prompt = f"""You are an expert evaluator comparing multiple AI model responses to the same question.

QUESTION:
{question.prompt}

"""

        # Add evaluation criteria if available
        if question.evaluation_criteria:
            prompt += f"""EVALUATION CRITERIA:
{question.evaluation_criteria}

"""

        # Add expected output if available
        if question.expected_output:
            prompt += f"""EXPECTED OUTPUT:
{question.expected_output}

"""

        # Add all anonymized responses
        prompt += "MODEL RESPONSES:\n\n"
        for entry in anonymized_responses:
            prompt += f"=== Model {entry['label']} ===\n"
            prompt += f"{entry['response_text']}\n"

            if entry['reasoning']:
                prompt += f"\nReasoning/Thinking:\n{entry['reasoning']}\n"

            prompt += "\n"

        # Add code execution results if available
        if code_execution_results:
            prompt += "\nCODE EXECUTION RESULTS:\n\n"
            for entry in anonymized_responses:
                model_name = entry['model_name']
                if model_name in code_execution_results:
                    result = code_execution_results[model_name]
                    prompt += f"=== Model {entry['label']} ===\n"
                    prompt += f"Passed: {result.get('passed', False)}\n"
                    if result.get('error'):
                        prompt += f"Error: {result['error']}\n"
                    if result.get('output'):
                        prompt += f"Output: {result['output']}\n"
                    prompt += "\n"

        # Add evaluation instructions
        prompt += """
EVALUATION TASK:
Compare all model responses against each other and the expected criteria. For each model, provide:
1. A score from 0-100 (higher is better)
2. Detailed reasoning explaining the score in the context of other responses

SCORING GUIDELINES:
- Consider ALL responses together - scores should reflect relative quality
- 90-100: Exceptional response, clearly superior to others
- 80-89: Excellent response, among the best
- 70-79: Good response, above average
- 60-69: Adequate response, meets basic requirements
- 50-59: Below average, has significant issues
- 0-49: Poor response, fails to meet requirements

Be harsh but fair. Consider:
- Correctness and completeness
- Code quality (if applicable): proper structure, error handling, edge cases
- Clarity and explanation quality
- Efficiency and best practices
- How well it addresses the specific requirements

OUTPUT FORMAT:
For each model, use this exact format:

Model [LABEL]:
SCORE: [0-100]
REASONING: [Your detailed reasoning comparing to other responses]

---

Provide evaluations for ALL models now:
"""

        return prompt

    async def _call_judge(self, prompt: str) -> str:
        """Call the judge model with the comparative prompt."""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = await self.client.achat(
                model=self.judge_model,
                messages=messages,
                temperature=0.0,  # Deterministic evaluation
                stream=False
            )

            return response.content

        except Exception as e:
            print(f"Error calling judge model: {e}")
            return ""

    def _parse_comparative_response(
        self,
        judge_response: str,
        anonymized_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse the judge's comparative response to extract scores and reasoning.

        Returns:
            Dict mapping model_name -> {score, reasoning, details}
        """
        results = {}

        # Create mapping from label to model_name
        label_to_model = {entry['label']: entry['model_name'] for entry in anonymized_responses}

        # Split by "Model X:" sections
        sections = re.split(r'(?=Model [A-Z]:|Model_\d+:)', judge_response)

        for section in sections:
            if not section.strip():
                continue

            # Extract model label
            label_match = re.match(r'Model ([A-Z]|_\d+):', section)
            if not label_match:
                continue

            label = label_match.group(1)
            if label not in label_to_model:
                continue

            model_name = label_to_model[label]

            # Extract score
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', section, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0

            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n---|\Z)', section, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            results[model_name] = {
                'score': score,
                'passed': score >= 70,  # Use same passing threshold as regular judge
                'reasoning': reasoning,
                'details': {
                    'full_judge_response': section.strip(),
                    'comparative_evaluation': True,
                    'anonymous_label': label
                }
            }

        return results
