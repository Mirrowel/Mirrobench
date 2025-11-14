# Custom OpenAI-Compatible Provider Usage

This document explains how to use the comprehensive custom OpenAI-compatible provider support in the LLM benchmark library.

## Overview

The library now supports adding custom OpenAI-compatible providers as first-class citizens with full plugin integration. You can configure any provider that follows the OpenAI API format by simply setting environment variables - no code changes required.

## Architecture

### Dynamic Plugin System
- Custom providers are automatically detected and given proper plugin instances
- Full integration with the existing provider plugin architecture
- Class-based configuration with `skip_cost_calculation` and other provider attributes
- Seamless LiteLLM integration while preserving original provider names

### Model Definitions Support
- Static model definitions via environment variables
- Model-specific options (like `reasoning_effort`)
- Fallback to dynamic discovery when available

## Configuration

### Basic Setup

To add a custom OpenAI-compatible provider, set these environment variables:

1. `PROVIDER_API_KEY` - Your API key for the provider
2. `PROVIDER_API_BASE` - The base URL for the provider's API
3. `PROVIDER_MODELS` - (Optional) Static model definitions in JSON format

Where `PROVIDER` is the name you want to give to your provider (in lowercase).

### Complete Example

```bash
# Basic configuration
OPencode_API_KEY=sk-your-opencode-key
OPencode_API_BASE=https://opencode.ai/zen/v1

# With static model definitions
OPencode_MODELS={"big-pickle": {"id": "big-pickle", "options": {"reasoning_effort": "high"}}}

# Multiple providers with full configuration
IFLOW_API_KEY=sk-iflow-key
IFLOW_API_BASE=https://apis.iflow.cn/v1
IFLOW_MODELS={"GLM-4.6": {"id": "glm-4.6", "options": {"reasoning_effort": "high"}}, "DS-R1": {"id": "deepseek-r1", "options": {"reasoning_effort": "high"}}}
```

### Model Definitions Format

```json
{
  "model_name": {
    "id": "api-model-id",
    "options": {
      "reasoning_effort": "high",
      "temperature": 0.1,
      "max_tokens": 4096
    }
  }
}
```

- `model_name`: Name you'll use in code (e.g., `provider/model_name`)
- `id`: Actual model ID the API expects
- `options`: Optional model-specific parameters

## Usage in Code

### Basic Usage

```python
from lib.rotator_library.client import RotatingClient

# Initialize client with custom provider keys
client = RotatingClient(api_keys={
    'opencode': ['sk-your-opencode-key'],
    'iflow': ['sk-iflow-key']
})

# Use models from custom providers
response = await client.acompletion(
    model="opencode/big-pickle",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

response = await client.acompletion(
    model="iflow/GLM-4.6",
    messages=[{"role": "user", "content": "How are you?"}]
)
```

### Streaming Support

```python
# Streaming works seamlessly with custom providers
async for chunk in client.astream(
    model="iflow/DS-R1",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Key Features

### 1. Automatic Provider Detection
- Scans environment for `*_API_BASE` patterns
- Creates dynamic plugin instances automatically
- Excludes known built-in providers to prevent conflicts

### 2. Full Plugin Integration
- Custom providers get proper plugin classes with `ProviderInterface`
- Class attributes like `skip_cost_calculation: bool = True`
- Seamless integration with usage manager and error handling

### 3. Model Management
- **Static Definitions**: Define models via `PROVIDER_MODELS` JSON
- **Dynamic Discovery**: Automatic `/models` endpoint fetching
- **Model Options**: Per-model configuration (reasoning effort, etc.)
- **Fallback**: Uses static + dynamic models together

### 4. Cost Calculation Handling
- Automatic cost calculation skipping for custom providers
- Usage tracking still works (prompt/completion tokens)
- No more "model not found" cost errors

### 5. LiteLLM Integration
- Internal conversion to `openai/` format for LiteLLM
- Original provider names preserved in all logs and tracking
- Model options automatically applied to API calls

### 6. Complete Feature Support
- Key rotation across multiple API keys
- Rate limiting and cooldown handling
- Error classification and intelligent retry logic
- Usage statistics and tracking
- Streaming and non-streaming calls
- Authentication and authorization handling

## Real-World Examples

### Example 1: OpenEncode Provider
```bash
# .env configuration
OPencode_API_KEY=sk-ZXqH6wpvwom4TLj3h4cr2iHdaP5tUoXFl7K9bn5FQ6ULGGX2BqEhRCKqaEhp3D4S
OPencode_API_BASE=https://opencode.ai/zen/v1
OPencode_MODELS={"big-pickle": {"id": "big-pickle", "options": {"reasoning_effort": "high"}}}
```

```python
# Usage
response = await client.acompletion(
    model="opencode/big-pickle",
    messages=[{"role": "user", "content": "Write a Python function"}],
    # reasoning_effort="high" automatically applied
)
```

### Example 2: iFlow Provider
```bash
# .env configuration
IFLOW_API_KEY=sk-9161574dba036059396d68c84821545f
IFLOW_API_BASE=https://apis.iflow.cn/v1
IFLOW_MODELS={
  "GLM-4.6": {"id": "glm-4.6", "options": {"reasoning_effort": "high"}},
  "Qwen3-Coder-Plus": {"id": "qwen3-coder-plus"},
  "DS-R1": {"id": "deepseek-r1", "options": {"reasoning_effort": "high"}},
  "K2": {"id": "kimi-k2"}
}
```

```python
# Usage with different models
models = ["iflow/GLM-4.6", "iflow/DS-R1", "iflow/K2"]
for model in models:
    response = await client.acompletion(
        model=model,
        messages=[{"role": "user", "content": "Hello"}]
    )
    # Model options applied automatically based on definition
```

### Example 3: Multiple Custom Providers
```bash
# Provider A
PROVIDERA_API_KEY=sk-key-a
PROVIDERA_API_BASE=https://api.provider-a.com/v1
PROVIDERA_MODELS={"gpt-4": {"id": "gpt-4"}}

# Provider B  
PROVIDERB_API_KEY=sk-key-b
PROVIDERB_API_BASE=https://api.provider-b.com/v1
PROVIDERB_MODELS={"claude-3": {"id": "claude-3-opus-20240229", "options": {"temperature": 0.1}}}
```

## Advanced Configuration

### Multiple API Keys per Provider
```bash
# Multiple keys for rotation (adds _1, _2, etc.)
OPencode_API_KEY_1=sk-first-key
OPencode_API_KEY_2=sk-second-key
OPencode_API_KEY_3=sk-third-key
OPencode_API_BASE=https://opencode.ai/zen/v1
```

### Model Options Reference
Common options that can be specified in model definitions:

```json
{
  "model_name": {
    "id": "api-model-id",
    "options": {
      "reasoning_effort": "low|medium|high",
      "temperature": 0.0-2.0,
      "max_tokens": 1-32768,
      "top_p": 0.0-1.0,
      "frequency_penalty": -2.0-2.0,
      "presence_penalty": -2.0-2.0
    }
  }
}
```

## Supported Providers

The library automatically excludes known built-in providers from being loaded as custom providers:
- openai, anthropic, google, gemini, nvidia, mistral, cohere, groq, openrouter, chutes

## Testing Your Setup

### Verify Provider Detection
```python
from lib.rotator_library.providers import PROVIDER_PLUGINS

print("Registered providers:")
for name, plugin_class in PROVIDER_PLUGINS.items():
    print(f"  {name}: {plugin_class}")
    if hasattr(plugin_class, 'skip_cost_calculation'):
        print(f"    -> skip_cost_calculation: {plugin_class.skip_cost_calculation}")
```

### Test API Calls
```python
import asyncio
from lib.rotator_library.client import RotatingClient

async def test():
    client = RotatingClient(api_keys={
        'yourprovider': ['sk-your-key']
    })
    
    response = await client.acompletion(
        model="yourprovider/your-model",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10
    )
    print(f"Success: {response.choices[0].message.content}")

asyncio.run(test())
```

## Troubleshooting

### Provider Not Detected
- Ensure environment variable ends with exactly `_API_BASE` (9 characters)
- Check provider name doesn't conflict with built-in providers
- Verify environment variables are loaded (use `load_dotenv()`)

### Cost Calculation Errors
- Custom providers automatically skip cost calculation
- If you see cost errors, ensure plugin is properly registered
- Check that `skip_cost_calculation: bool = True` is set

### Model Options Not Applied
- Verify JSON format in `PROVIDER_MODELS` is valid
- Check model name matches definition key
- Ensure options are in the correct `options` object

### API Calls Fail
- Verify `API_BASE` URL is accessible
- Check API key validity
- Ensure provider follows OpenAI API format exactly
- Test with curl first to verify API works

### Dynamic Model Discovery Issues
- Some providers don't implement `/models` endpoint
- Use static model definitions as fallback
- Check provider API documentation

## Migration from Previous Versions

If you were using an older version with custom provider handling:

1. **No code changes needed** - existing configurations continue to work
2. **Add model definitions** - optional but recommended for better model management
3. **Cost calculation** - now automatically skipped, no more errors
4. **Plugin benefits** - full integration with provider ecosystem

## Example: Complete Production Setup

```bash
# .env file
# Custom providers with full configuration
OPencode_API_KEY=sk-your-opencode-key
OPencode_API_BASE=https://opencode.ai/zen/v1
OPencode_MODELS={"big-pickle": {"id": "big-pickle", "options": {"reasoning_effort": "high"}}}

IFLOW_API_KEY=sk-your-iflow-key  
IFLOW_API_BASE=https://apis.iflow.cn/v1
IFLOW_MODELS={"GLM-4.6": {"id": "glm-4.6", "options": {"reasoning_effort": "high"}}, "DS-R1": {"id": "deepseek-r1", "options": {"reasoning_effort": "high"}}}

# Built-in providers still work
OPENAI_API_KEY=sk-openai-key
ANTHROPIC_API_KEY=sk-anthropic-key
```

```python
# Python usage - everything works seamlessly
from lib.rotator_library.client import RotatingClient

client = RotatingClient(api_keys={
    'opencode': ['sk-opencode-key'],
    'iflow': ['sk-iflow-key'],
    'openai': ['sk-openai-key'],
    'anthropic': ['sk-anthropic-key']
})

# All providers work the same way
models = [
    'opencode/big-pickle',      # Custom with options
    'iflow/GLM-4.6',          # Custom with options  
    'iflow/DS-R1',             # Custom with reasoning effort
    'openai/gpt-4',           # Built-in
    'anthropic/claude-3-5-sonnet' # Built-in
]

for model in models:
    response = await client.acompletion(
        model=model,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(f"{model}: {response.choices[0].message.content}")
```

This comprehensive system makes custom OpenAI-compatible providers work exactly like built-in providers, with full support for model definitions, options, cost handling, and all library features.

### Example 2: Multiple Custom Providers

```bash
# First custom provider
PROVIDER1_API_KEY=sk-key-1
PROVIDER1_API_BASE=https://api.provider1.com/v1

# Second custom provider
PROVIDER2_API_KEY=sk-key-2
PROVIDER2_API_BASE=https://api.provider2.com/v1

# Third provider with descriptive name
MYLLM_API_KEY=sk-key-3
MYLLM_API_BASE=https://api.myllm.com/v1
```

## Usage in Code

Once configured, you can use custom providers just like any other provider:

```python
from lib.rotator_library.client import RotatingClient

# Initialize client with custom provider keys
client = RotatingClient(api_keys={
    'customprovider': ['sk-your-custom-api-key'],
    'provider1': ['sk-key-1'],
    'myllm': ['sk-key-3']
})

# Use models from custom providers
response = await client.acompletion(
    model="customprovider/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

response = await client.acompletion(
    model="provider1/custom-model-name",
    messages=[{"role": "user", "content": "How are you?"}]
)
```

## Key Features

### 1. Automatic Provider Detection
- Custom providers are automatically detected from environment variables
- No code changes required to add new providers

### 2. Full Library Support
- Key rotation and management
- Rate limiting handling
- Error classification and retry logic
- Usage tracking and cost calculation
- Streaming support

### 3. Model Discovery
- Custom providers automatically expose their available models
- Models are prefixed with the provider name (e.g., `customprovider/model-name`)

### 4. OpenAI-Compatible
- Works with any provider that follows the OpenAI API format
- Supports standard completion and embedding endpoints

## Supported Providers

The library automatically excludes known providers from being loaded as custom providers:
- openai
- anthropic
- google
- gemini
- nvidia
- mistral
- cohere
- groq
- openrouter

## Testing

To test your custom provider configuration:

```bash
python test_custom_providers.py
```

This will verify that:
- Custom providers are correctly detected
- API base URLs are properly configured
- Model names are handled correctly
- Multiple providers work simultaneously

## Troubleshooting

### Provider Not Detected
- Ensure the environment variable ends with `_API_BASE`
- Check that the provider name doesn't conflict with known providers
- Verify the environment variables are set correctly

### API Calls Fail
- Verify the `API_BASE` URL is correct and accessible
- Check that your API key is valid
- Ensure the provider follows the OpenAI API format

### Models Not Available
- Some providers may not implement the `/models` endpoint
- You can still use models by specifying them directly in the format: `provider/model-name`

## Example Real-World Providers

### Together AI
```bash
TOGETHER_API_KEY=your-together-api-key
TOGETHER_API_BASE=https://api.together.xyz/v1
```

### Perplexity
```bash
PERPLEXITY_API_KEY=your-perplexity-api-key
PERPLEXITY_API_BASE=https://api.perplexity.ai
```

### Any OpenAI-Compatible Provider
```bash
YOURPROVIDER_API_KEY=your-api-key
YOURPROVIDER_API_BASE=https://your-provider.com/api/v1
```

---

## Advanced Configuration Features

### Model Configurations in config.yaml

In addition to environment variables, you can configure model-specific settings directly in `config.yaml` using the `model_configs` section. This allows you to set system instructions and additional API options for any model.

#### Basic Structure

```yaml
model_configs:
  "provider/model-name":
    system_instruction: "Custom instruction for this model"
    options:
      api_parameter: value
```

#### Features

1. **System Instructions**: Prepended to prompts for specific models
2. **API Options**: Additional parameters merged into API requests
3. **Applies to**:
   - All benchmark models
   - Judge model (LLM-as-judge evaluator)
   - Fixer model (code formatting)

#### Example: Reasoning Effort for Custom Providers

```yaml
model_configs:
  # Custom provider with reasoning effort
  "opencode/big-pickle":
    system_instruction: "detailed thinking on"
    options:
      reasoning_effort: "high"

  # Multiple models with same options
  "iflow/DS-R1":
    options:
      reasoning_effort: "high"

  # Judge model configuration
  "gemini/gemini-2.5-pro":
    options:
      reasoning_effort: "high"
```

The API request will automatically include these options:
```json
{
  "model": "opencode/big-pickle",
  "messages": [...],
  "reasoning_effort": "high"
}
```

### Per-Provider Concurrency Limits

Different providers have different rate limits. Configure per-provider concurrency in `config.yaml`:

```yaml
# Global default for all providers
max_concurrent: 10

# Per-provider limits
provider_concurrency:
  opencode: 2      # Custom provider with low limits
  iflow: 5         # Another custom provider
  gemini: 20       # Gemini can handle high concurrency
  openai: 5        # OpenAI has strict rate limits
```

#### How It Works

1. **Provider Detection**: Automatically extracted from model name
   - `opencode/big-pickle` → provider: `opencode`
   - `iflow/DS-R1` → provider: `iflow`

2. **Concurrency Application**: Each provider gets its own limit
   - If provider not specified, uses global `max_concurrent`
   - Prevents rate limit errors
   - Optimizes throughput

#### Complete Example

```yaml
# config.yaml
max_concurrent: 10

provider_concurrency:
  opencode: 2      # Conservative for custom provider
  iflow: 5         # Moderate for iFlow
  gemini: 20       # Aggressive for Gemini

models:
  - "opencode/big-pickle"      # Uses 2 concurrent requests
  - "iflow/DS-R1"              # Uses 5 concurrent requests
  - "gemini/gemini-2.5-pro"    # Uses 20 concurrent requests
  - "anthropic/claude-3-5-sonnet"  # Uses 10 (global default)

judge_model: "gemini/gemini-2.5-pro"

model_configs:
  "opencode/big-pickle":
    system_instruction: "detailed thinking on"
    options:
      reasoning_effort: "high"

  "iflow/DS-R1":
    options:
      reasoning_effort: "high"

  "gemini/gemini-2.5-pro":
    options:
      reasoning_effort: "high"
```

### All Configuration Methods Combined

You can use all three configuration methods together:

1. **Environment Variables** (from this guide):
   ```bash
   OPencode_API_KEY=sk-key
   OPencode_API_BASE=https://opencode.ai/zen/v1
   OPencode_MODELS={"big-pickle": {"id": "big-pickle"}}
   ```

2. **Model Configs** (in config.yaml):
   ```yaml
   model_configs:
     "opencode/big-pickle":
       system_instruction: "detailed thinking on"
       options:
         reasoning_effort: "high"
   ```

3. **Provider Concurrency** (in config.yaml):
   ```yaml
   provider_concurrency:
     opencode: 2
   ```

### Migration Notes

#### From Environment Variables to config.yaml

Previously, model options were set in `PROVIDER_MODELS`:
```bash
OPencode_MODELS={"big-pickle": {"id": "big-pickle", "options": {"reasoning_effort": "high"}}}
```

You can now also configure in `config.yaml` (both work):
```yaml
model_configs:
  "opencode/big-pickle":
    options:
      reasoning_effort: "high"
```

**Recommendation**:
- Use environment variables for provider connection (API keys, base URLs, model IDs)
- Use `config.yaml` for model behavior (system instructions, API options, concurrency)

### Tips for Custom Providers

1. **Start with low concurrency** and increase gradually:
   ```yaml
   provider_concurrency:
     yourcustomprovider: 2  # Start conservative
   ```

2. **Use system instructions** for models without native reasoning:
   ```yaml
   model_configs:
     "yourcustomprovider/model":
       system_instruction: "Think step by step before answering."
   ```

3. **Configure judge and fixer** if using custom providers:
   ```yaml
   judge_model: "opencode/big-pickle"
   fixer_model: "iflow/DS-R1"

   model_configs:
     "opencode/big-pickle":
       options:
         reasoning_effort: "high"
   ```

4. **Monitor rate limits** in the benchmark output:
   ```
   Using provider-specific concurrency: 2 for opencode
   ```

For more details on model configurations and per-provider concurrency, see the main `config.yaml` file for examples and documentation.