"""
Code fixer that uses LLM to reformat responses into correct format.
Useful when models provide good code but in wrong format.
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from lib.rotator_library.client import RotatingClient
from src.schemas import Question, ModelResponse


class CodeFixer:
    """Use LLM to fix code formatting in model responses."""

    def __init__(
        self,
        client: RotatingClient,
        fixer_model: str = "anthropic/claude-3-5-sonnet-20241022",
        model_options: Optional[Dict[str, Any]] = None
    ):
        self.client = client
        self.fixer_model = fixer_model
        self.model_options = model_options or {}

    async def fix_response(self, question: Question, response: ModelResponse) -> ModelResponse:
        """
        Use LLM to reformat a response into correct format.

        Args:
            question: The original question
            response: The model's response to fix

        Returns:
            ModelResponse: Fixed version with corrected formatting
        """
        # Build the fix prompt
        fix_prompt = self._build_fix_prompt(question, response)

        try:
            # Build request kwargs
            kwargs = {
                "model": self.fixer_model,
                "messages": [
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ],
                "temperature": 0.0  # Deterministic reformatting
            }

            # Add model-specific options if configured for fixer model
            if self.model_options:
                kwargs.update(self.model_options)

            # Call the fixer model
            fixer_response = await self.client.acompletion(**kwargs)

            # Extract the fixed response text
            if fixer_response and hasattr(fixer_response, 'choices'):
                fixed_text = fixer_response.choices[0].message.content
            else:
                fixed_text = str(fixer_response)

            # Create new response with fixed text
            fixed_response = ModelResponse(
                question_id=response.question_id,
                model_name=response.model_name,
                response_text=fixed_text,
                reasoning_content=response.reasoning_content,
                tool_calls=response.tool_calls,
                full_response=response.full_response,
                metrics=response.metrics,
                timestamp=datetime.now().isoformat(),
                error=None
            )

            return fixed_response

        except Exception as e:
            # If fix fails, return original with error note
            return ModelResponse(
                question_id=response.question_id,
                model_name=response.model_name,
                response_text=response.response_text,
                reasoning_content=response.reasoning_content,
                tool_calls=response.tool_calls,
                full_response=response.full_response,
                metrics=response.metrics,
                timestamp=datetime.now().isoformat(),
                error=f"Fix failed: {str(e)}"
            )

    def _build_fix_prompt(self, question: Question, response: ModelResponse) -> str:
        """Build the prompt for the fixer model."""

        # Determine the type of fix needed
        is_multi_file = (
            question.metadata.get("requires_multi_file", False) or
            question.metadata.get("expected_files", [])
        )

        if is_multi_file:
            format_instructions = """Format the code as separate files using this syntax:

```html:index.html
[HTML code here]
```

```css:styles.css
[CSS code here]
```

```javascript:app.js
[JavaScript code here]
```

Make sure:
1. Each file has the correct language tag and filename
2. Files reference each other correctly (e.g., <link href="styles.css">, <script src="app.js">)
3. Code is properly organized by file type"""
        else:
            format_instructions = """Format the code using standard markdown code blocks:

```python
[Python code here]
```

or

```html
[HTML code here]
```

Make sure:
1. Code has the correct language tag
2. Code is complete and runnable
3. All necessary parts are included"""

        prompt = f"""You are a code formatting assistant. A model provided a response to a coding question, but the formatting is incorrect or unclear.

**Original Question:**
{question.prompt}

**Model's Response (with formatting issues):**
{response.response_text}

**Your Task:**
Reformat this response so the code is properly structured and can be evaluated. Do not change the logic or functionality - only fix the formatting.

{format_instructions}

**Important:**
- Preserve all the original code logic
- Only fix formatting, markdown syntax, and file organization
- Do not add new features or change behavior
- Do NOT add explanations, descriptions, or commentary - output ONLY the code
- If there are explanations in the original, remove them and keep only the code
- If the code is already well-formatted, return it as-is with proper markdown

Provide the reformatted response (code only, no explanations):"""

        return prompt


def format_fixed_filename(original_filename: str) -> str:
    """
    Generate filename for fixed response.

    Args:
        original_filename: e.g., "question_id.json"

    Returns:
        Fixed filename: e.g., "question_id_fixed.json"
    """
    if original_filename.endswith('.json'):
        return original_filename.replace('.json', '_fixed.json')
    return original_filename + '_fixed'
