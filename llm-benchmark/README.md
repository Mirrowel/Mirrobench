# 🔴 MirroBench - Comprehensive LLM Benchmark System [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Vue.js](https://img.shields.io/badge/vue.js-3.x-green.svg)](https://vuejs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-red.svg)](https://fastapi.tiangolo.com)

**MirroBench** is a comprehensive web-based benchmarking platform for evaluating Large Language Models across quality, speed, cost, and capabilities. The system tests models on 27 comprehensive project-level questions spanning CLI tools, games, web applications, visualizations, simulations, and creative coding.

> **Note**: This is now primarily a **web-based system**. While `run.py` still exists for legacy use, all modern operations are managed through the web viewer interface.

## 🚀 Quick Start

### 3-Minute Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd llm-benchmark
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Start web interface
python viewer/server.py
# Open http://localhost:8000
```

That's it! You're ready to benchmark LLMs with a modern web interface.

### What You Can Do Immediately
- ✅ **Run Benchmarks**: Click "Run Benchmark" tab, add models like `openai/gpt-4`, hit Start
- ✅ **View Results**: Interactive leaderboard with scores, costs, speeds
- ✅ **See Generated Apps**: View HTML/Canvas games and apps actually running
- ✅ **Compare Models**: Side-by-side comparisons with visual diff

**No config file editing required** - the web interface handles everything!

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Question Catalog](#question-catalog)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Evaluation Methods](#evaluation-methods)
- [Web Interface](#web-interface)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

MirroBench is designed to be the definitive LLM evaluation platform, combining:

- **Comprehensive Testing**: 27 project-level questions across 6 categories
- **Multiple Evaluation Methods**: LLM-as-judge, code execution, tool validation, comparative analysis
- **Modern Web Interface**: Vue.js + FastAPI with real-time progress tracking
- **Cost & Performance Tracking**: Token usage, latency, TTFT, TPS metrics
- **Flexible Configuration**: Support for 20+ LLM providers with custom provider setup
- **Professional Results**: Interactive leaderboards with artifact display

### What Makes MirroBench Different

Unlike simple Q&A benchmarks, MirroBench evaluates models on **real-world programming tasks** that require:

- Complete application development
- Multi-file organization
- Complex problem-solving
- Code quality and architecture
- User interface design
- Performance optimization

## 🏗️ Architecture

### Modern Web-Based Design

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Vue.js 3 Frontend                        │ │
│  │  • Interactive Leaderboard                         │ │
│  │  • Visual Config Editor                           │ │
│  │  • Real-time Progress Tracking                    │ │
│  │  • Artifact Viewer (iframes)                     │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                                 │
                                 ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────┐
│                FastAPI Server                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Core Services                             │ │
│  │  • Benchmark Job Manager                           │ │
│  │  • Results API                                     │ │
│  │  • Configuration Management                        │ │
│  │  • Artifact Extraction & Display                  │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────┐
│                Benchmark Engine                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Evaluation Pipeline                       │ │
│  │  • LLM-as-Judge                                    │ │
│  │  • Code Execution                                  │ │
│  │  • Tool Validation                                 │ │
│  │  • Cost Calculation                                │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Modules

#### Benchmark Engine (`src/`)
- **`runner.py`**: Main orchestrator for benchmark execution
- **`config_loader.py`**: Configuration management and validation
- **`question_loader.py`**: Question loading and filtering
- **`results_manager.py`**: Results storage and retrieval
- **`cost_calculator.py`**: API cost calculation and tracking
- **`evaluators/`**: Multiple evaluation methods
  - `llm_judge.py`: LLM-as-judge evaluation
  - `code_executor.py`: Code execution and validation
  - `tool_validator.py`: Function calling accuracy
  - `comparative_judge.py`: Side-by-side model comparison

#### Web Interface (`viewer/`)
- **`server.py`**: FastAPI backend with REST APIs
- **`templates/`**: HTML templates with Vue.js 3
- **`static/`**: CSS, JavaScript, and assets
- **Backend modules**: Async benchmark execution, job management

## ✨ Features

### Benchmark Capabilities

- **27 Project-Level Questions**: Real-world programming challenges
- **6 Categories**: CLI Tools, Games, Web Apps, Visualizations, Simulations, Creative Coding
- **Multiple Evaluation Types**: LLM Judge, Code Execution, Tool Calling, Exact Match, Contains, Comparative
- **Custom Provider Support**: Add any OpenAI-compatible API
- **Concurrent Execution**: Configurable concurrency per provider
- **Real-time Progress**: Live updates during benchmark execution

### Web Interface

- **Modern Vue.js 3 UI**: Reactive, responsive design
- **Multiple Evaluation Modes**: Individual, Comparative, Human Judge, Author's Choice
- **Interactive Leaderboards**: Sort, filter, and compare results
- **Artifact Display**: View and interact with generated code
- **Configuration Editor**: Visual and YAML configuration editing
- **Real-time Logs**: Live benchmark execution monitoring

### Analytics & Tracking

- **Performance Metrics**: TTFT, TPS, latency, token counts
- **Cost Calculation**: Per-model and total cost tracking
- **Cost Efficiency**: Quality per dollar analysis
- **Response Versioning**: Track changes and fixes
- **Human Ratings**: Add human evaluation scores
- **Export Capabilities**: Results export and reporting

## 🛠️ Installation

### System Requirements

- **Python**: 3.8+ (3.11+ recommended)
- **Node.js**: Required for JavaScript execution questions
- **Git**: For cloning and version control
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space for results

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone <your-repo-url>
cd llm-benchmark
```

#### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install custom library
pip install -e lib/rotator_library

# Verify installation
python verify_setup.py
```

#### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor
```

#### 4. Start Application
```bash
# Start web interface
python viewer/server.py

# Open browser to http://localhost:8000
```

### Production Deployment

#### Docker Setup
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python", "viewer/server.py"]
```

#### Systemd Service
```ini
[Unit]
Description=MirroBench LLM Benchmark
After=network.target

[Service]
Type=simple
User=benchmark
WorkingDirectory=/opt/mirrorbench
ExecStart=/opt/mirrorbench/venv/bin/python viewer/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your API keys:

```bash
# Standard Providers
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
NVIDIA_NIM_API_KEY=nvapi-...

# Custom Providers (Pattern: {PROVIDER}_API_KEY + {PROVIDER}_API_BASE)
OPencode_API_KEY=sk-your-key
OPencode_API_BASE=https://opencode.ai/zen/v1
```

### Configuration File

Copy `config_template.yaml` to `config.yaml` and customize:

```yaml
# Models to benchmark
models:
  - "openai/gpt-4"
  - "anthropic/claude-3-5-sonnet"
  - "gemini/gemini-2.0-flash-exp"

# Judge model for evaluation
judge_model: "anthropic/claude-3-5-sonnet-20241022"

# Categories to test (empty = all)
categories: ["cli_tools", "web_apps"]

# Concurrency settings
max_concurrent: 10
provider_concurrency:
  openai: 5
  anthropic: 10
  gemini: 20

# Model-specific configurations
model_configs:
  "openai/o1":
    system_instruction: "Think step by step before answering."
    options:
      reasoning_effort: "high"
      temperature: 0.7
```

### Custom Provider Setup

Add any OpenAI-compatible provider:

```bash
# Environment variables
YOURPROVIDER_API_KEY=sk-your-api-key
YOURPROVIDER_API_BASE=https://api.yourprovider.com/v1

# Optional: Model definitions
YOURPROVIDER_MODELS={
  "model-name": {
    "id": "model-name",
    "options": {
      "temperature": 0.7,
      "max_tokens": 4000
    }
  }
}
```

```yaml
# Configuration
models:
  - "yourprovider/model-name"

model_configs:
  "yourprovider/model-name":
    system_instruction: "Custom instructions"
    options:
      temperature: 0.5
```

## 📚 Question Catalog

MirroBench contains **27 comprehensive project-level questions** across **6 categories**. For detailed specifications, prompts, and evaluation criteria, see the [QUESTIONS.md](QUESTIONS.md) file.

### Quick Overview

| Category | Questions | Focus | Difficulty |
|----------|-----------|-------|------------|
| **CLI Tools** | 5 | Terminal applications | Medium - Very Hard |
| **Games** | 10 | Interactive games (Python + Web) | Medium - Very Hard |
| **Web Apps** | 4 | Full-featured web applications | Medium - Very Hard |
| **Visualizations** | 4 | Graphics and data visualization | Medium - Very Hard |
| **Simulations** | 5 | Physics and scientific simulations | Very Hard |
| **Creative Coding** | 6 | Advanced creative applications | Hard - Very Hard |

### Evaluation Methods

- **Code Execution** (10 questions): Automated testing of functionality
- **LLM Judge** (14 questions): Qualitative assessment for creative/complex tasks
- **Tool Validation** (1 question): Specialized validation for specific requirements

### Key Features

- **Real-world complexity**: Complete applications, not toy problems
- **Multi-language support**: Python, JavaScript/Node.js, Rust, HTML/CSS/JS
- **Modern technologies**: Pygame, Canvas, WebGL, Web Audio, Curses
- **Comprehensive evaluation**: Functionality, quality, UX, creativity

> **See [QUESTIONS.md](QUESTIONS.md)** for the complete catalog with detailed documentation for each category.

## 📖 Usage Guide

### Web Interface Workflow

#### 1. Configuration
- Access Settings tab to configure models and categories
- Use visual editor or YAML editor
- Validate and save configuration

#### 2. Run Benchmarks
- Navigate to "Run Benchmark" tab
- Select models, categories, and questions
- Configure concurrency and timeouts
- Start benchmark with real-time progress

#### 3. View Results
- **Individual Tab**: Per-model scores and detailed responses
- **Comparative Tab**: Side-by-side model comparisons
- **Human Judge Tab**: Add human evaluations
- **Author's Choice Tab**: Expert rankings

#### 4. Analyze Performance
- View leaderboards with multiple sorting options
- Examine detailed responses and code artifacts
- Compare cost efficiency metrics
- Export results for reporting

### Command Line Usage

#### Basic Benchmark
```bash
# Use default configuration
python run.py

# Use custom configuration
python run.py --config custom.yaml

# Run specific categories
python run.py --categories cli_tools,web_apps
```

#### Advanced Options
```bash
# Custom concurrency
python run.py --max-concurrent 20

# Specific questions
python run.py --question-ids cli_terminal_text_editor,webapp_drawing_app

# Debug mode
python run.py --debug
```

### API Usage

#### Start Benchmark
```bash
curl -X POST "http://localhost:8000/api/benchmark/start" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["openai/gpt-4", "anthropic/claude-3-5-sonnet"],
    "categories": ["cli_tools"],
    "max_concurrent": 10
  }'
```

#### Get Results
```bash
curl "http://localhost:8000/api/leaderboard/unified"
```

## 🔌 API Reference

### Benchmark Endpoints

#### POST `/api/benchmark/start`
Start a new benchmark run
```json
{
  "models": ["openai/gpt-4"],
  "categories": ["cli_tools"],
  "question_ids": [],
  "max_concurrent": 10,
  "provider_concurrency": {"openai": 5}
}
```

#### GET `/api/benchmark/status`
Get current benchmark status
```json
{
  "status": "running",
  "job": {
    "job_id": "benchmark_20231201_120000",
    "progress": {...}
  }
}
```

#### DELETE `/api/benchmark/stop`
Stop running benchmark

### Results Endpoints

#### GET `/api/leaderboard/unified`
Get unified leaderboard across all runs
```json
{
  "leaderboard": [
    {
      "model_name": "openai/gpt-4",
      "overall_score": 85.2,
      "category_scores": {"cli_tools": 88.5},
      "total_cost": 12.34,
      "cost_efficiency_score": 6.91
    }
  ]
}
```

#### GET `/api/runs/{run_id}/models/{model}/questions/{question_id}`
Get specific response with evaluations
```json
{
  "question": {...},
  "response": {...},
  "evaluation": {...},
  "artifact": "extracted_code",
  "has_fixed_version": false
}
```

### Configuration Endpoints

#### GET `/api/config`
Get current configuration
```json
{
  "content": "models:\n  - openai/gpt-4\n..."
}
```

#### POST `/api/config/validate`
Validate configuration without saving
```json
{
  "yaml_content": "models:\n  - openai/gpt-4",
  "valid": true,
  "errors": []
}
```

#### POST `/api/config/save`
Save configuration with backup
```json
{
  "yaml_content": "models:\n  - openai/gpt-4",
  "success": true,
  "backup_path": "config.yaml.backup.20231201_120000"
}
```

### Comparative Judge Endpoints

#### POST `/api/comparative-judge/start`
Start comparative evaluation
```json
{
  "run_ids": ["run1", "run2"],
  "question_ids": ["q1", "q2"]
}
```

#### GET `/api/comparative-judge/jobs/{job_id}/results`
Get comparative results
```json
{
  "results": {
    "q1": {
      "openai/gpt-4": {"score": 85, "reasoning": "..."},
      "anthropic/claude": {"score": 78, "reasoning": "..."}
    }
  }
}
```

## 🧪 Evaluation Methods

### 1. LLM Judge Evaluation

**Purpose**: Subjective quality assessment by a powerful LLM
**Use Case**: Complex tasks requiring human-like judgment
**Process**:
1. Judge model receives question, response, and code execution results
2. Evaluates based on criteria in question metadata
3. Returns score (0-100), pass/fail, and detailed reasoning
4. Scoring guidelines:
   - 100: Perfect, beautiful, exceeds expectations
   - 90-99: Excellent with minor imperfections
   - 80-89: Good but lacks polish
   - 70-79: Adequate with notable issues
   - 60-69: Partially works with significant issues
   - 40-59: Poor with major errors
   - 20-39: Severely flawed
   - 0-19: Completely wrong

### 2. Code Execution Evaluation

**Purpose**: Technical validation of generated code
**Use Case**: Programming tasks with functional requirements
**Process**:
1. Extract code from response using multiple patterns
2. Execute in isolated environment with timeout
3. Validate output against expected results
4. For multi-file apps: check structure, linking, and organization
5. Return score based on functionality and correctness

**Supported Languages**:
- **Python**: Executes in temporary directory
- **JavaScript**: Uses Node.js
- **HTML**: Validates structure and rendering
- **Multi-file**: Validates file organization and linking

### 3. Tool Calling Validation

**Purpose**: Function calling accuracy assessment
**Use Case**: Questions requiring tool/API usage
**Process**:
1. Extract expected tool calls from question metadata
2. Parse actual tool calls from model response
3. Match calls by function name
4. Compare arguments with flexible matching
5. Calculate score: (matched_calls / total_expected) * 100

**Features**:
- Case-insensitive string comparison
- Type conversion support
- Exact match for non-strings
- Detailed reasoning for mismatches

### 4. Exact Match & Contains

**Purpose**: String-based validation
**Use Case**: Questions with specific expected outputs
**Process**:
- **Exact Match**: Compare response directly to expected output
- **Contains**: Check if expected substring exists in response

### 5. Comparative Judge

**Purpose**: Side-by-side model comparison
**Use Case**: Relative quality assessment across models
**Process**:
1. Anonymize and shuffle responses (Model A, B, C...)
2. Judge evaluates comparatively, not in isolation
3. Returns relative scores with comparison reasoning
4. Enables identification of best model for specific tasks

## 🌐 Web Interface

### Main Navigation

#### Individual Tab
- **Leaderboard**: Overall and per-category scores
- **Model Details**: Expandable model information
- **Response Viewer**: Detailed response examination
- **Artifact Display**: Interactive code execution
- **Metrics**: Performance and cost analytics

#### Comparative Tab
- **Job Creation**: Select runs and questions for comparison
- **Progress Tracking**: Real-time job status updates
- **Results Viewer**: Side-by-side model comparisons
- **Job History**: Previous comparative evaluations

#### Human Judge Tab
- **Rating Interface**: Slider-based scoring (0-100)
- **Comment System**: Detailed evaluation notes
- **Leaderboard**: Human-rated model rankings
- **Unified View**: Cross-run human evaluation aggregation

#### Author's Choice Tab
- **Drag-and-Drop Ranking**: Visual model ordering
- **Expert Evaluations**: Authoritative model assessments
- **Ranking Persistence**: Saved expert opinions
- **Public Rankings**: Shareable model comparisons

#### Run Benchmark Tab
- **Configuration Panel**: Model and category selection
- **Execution Panel**: Real-time progress and logs
- **Results Panel**: Live-updating results table
- **History Panel**: Previous benchmark runs

### Configuration Management

#### Visual Editor
- **Form-based Configuration**: Intuitive settings interface
- **Model Management**: Add/remove/configure models
- **Category Selection**: Checkbox-based question filtering
- **Concurrency Settings**: Provider-specific limits
- **Validation**: Real-time configuration checking

#### YAML Editor
- **CodeMirror Integration**: Syntax highlighting and validation
- **Bidirectional Sync**: Automatic synchronization with visual editor
- **Comment Preservation**: Maintain existing comments
- **Error Highlighting**: Real-time YAML validation

### Real-time Features

#### Progress Tracking
- **Live Updates**: WebSocket-based progress streaming
- **Phase Indicators**: Current benchmark phase display
- **Performance Metrics**: TTFT, TPS, latency tracking
- **Log Streaming**: Real-time log output with auto-scroll

#### Pop-out Windows
- **Dedicated Log Viewer**: Separate window for logs
- **Progress Monitoring**: Independent progress tracking
- **Communication**: PostMessage API for window interaction

## 🔧 Troubleshooting

### Common Issues

#### Setup Problems

**"No questions found" Error**
```bash
# Verify directory structure
ls -la questions/

# Check JSON validity
python -m json.tool questions/cli_tools/productivity_tools.json

# Verify configuration
grep questions_dir config.yaml
```

**"No API keys found" Error**
```bash
# Create .env file
cp .env.example .env

# Verify format
cat .env | grep API_KEY

# Test loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(dict(os.environ))"
```

**"rotator_library not installed" Error**
```bash
# Reinstall custom library
pip uninstall rotating-api-key-client
pip install -e lib/rotator_library

# Verify installation
python -c "from lib.rotator_library.client import RotatingClient; print('OK')"
```

#### Runtime Issues

**Slow Benchmark Execution**
```yaml
# Increase concurrency (watch rate limits)
max_concurrent: 20

# Use faster judge model
judge_model: "openai/gpt-3.5-turbo"

# Reduce question set for testing
categories:
  - cli_tools
```

**Memory Issues**
```yaml
# Decrease concurrent requests
max_concurrent: 5

# Run smaller batches
question_ids:
  - cli_terminal_text_editor
  - webapp_drawing_app
```

**Code Execution Failures**
```bash
# Check Python installation
python --version
python -c "print('Hello World')"

# Check Node.js installation
node --version
node -e "console.log('Hello World')"
```

#### Custom Provider Issues

```bash
# Test API directly
curl -H "Authorization: Bearer $YOURPROVIDER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"model-name","messages":[{"role":"user","content":"test"}]}' \
     $YOURPROVIDER_API_BASE/chat/completions

# Check environment variables
env | grep YOURPROVIDER
```

### Debug Mode

Enable verbose logging:
```bash
# Environment variable
export MIRRORBENCH_DEBUG=1

# Or in .env
echo "DEBUG=true" >> .env
```

### Log Analysis

#### Benchmark Logs
```bash
# View recent results
ls -la results/
tail -f results/latest/run.log

# Check for errors
grep -i error results/latest/run.log
grep -i timeout results/latest/run.log
```

#### Web Server Logs
```bash
# Server output shows API calls and errors
python viewer/server.py

# Check browser console for frontend issues
# Network tab for API failures
```

## 🤝 Contributing

We welcome contributions to MirroBench! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone <your-fork-url>
cd llm-benchmark

# Create development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e lib/rotator_library

# Install development dependencies
pip install -r requirements-dev.txt  # if available
```

### Contribution Areas

#### 1. Questions
- Add new benchmark questions
- Improve existing questions
- Add new categories
- Enhance evaluation criteria

#### 2. Evaluators
- Implement new evaluation methods
- Improve existing evaluators
- Add support for new languages
- Enhance accuracy metrics

#### 3. Web Interface
- Improve UI/UX design
- Add new features and visualizations
- Enhance performance
- Fix bugs and issues

#### 4. Core Engine
- Optimize performance
- Add new providers
- Improve error handling
- Enhance configuration options

### Submission Guidelines

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** and add tests if applicable
3. **Update documentation** for any new features
4. **Test thoroughly** including the verification script
5. **Submit a pull request** with clear description

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use ES6+, Vue.js 3 composition API
- **Documentation**: Update relevant docs for changes
- **Testing**: Ensure `verify_setup.py` passes

### Question Contribution Template

```json
{
  "id": "unique_question_id",
  "prompt": "Clear, detailed prompt",
  "system_prompt": "Optional system instruction",
  "expected_output": "Expected result if applicable",
  "evaluation_type": "llm_judge|code_execution|tool_calling|exact_match|contains",
  "evaluation_criteria": "How to evaluate the response",
  "category": "category_name",
  "subcategory": "subcategory_name",
  "tags": ["relevant", "tags"],
  "metadata": {
    "difficulty": "medium|hard|very_hard",
    "estimated_tokens": 1500,
    "should_be_complete_app": true
  }
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LiteLLM**: For unified LLM API access
- **Vue.js**: For the modern reactive frontend
- **FastAPI**: For the high-performance web backend
- **Rich**: For beautiful terminal output
- **Pydantic**: For data validation and settings
- All contributors and community members

## 📞 Support

- **Documentation**: Check this README and `/docs` directory
- **Questions**: See [QUESTIONS.md](QUESTIONS.md) for detailed question catalog
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Verification**: Run `python verify_setup.py` for setup issues

---

**MirroBench** - The definitive platform for comprehensive LLM evaluation. Built by the community, for the community.

*Last updated: November 2024*