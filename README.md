# LLM-Tools [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)

A collection of tools and utilities for working with Large Language Models (LLMs), designed to help developers test, benchmark, and optimize their AI applications.

## Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [LLM Benchmark](#llm-benchmark)
- [Getting Started](#getting-started)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

## Overview

This repository contains a curated collection of tools for LLM development and testing. Each tool is designed to solve specific challenges in working with language models, from performance benchmarking to API management and automation.

## Projects

### LLM Benchmark

A comprehensive benchmarking tool for evaluating and comparing LLM performance across different models and providers.

**Location**: `llm-benchmark/`

**Features**:
- **Automated Testing**: Run standardized test questions against multiple LLM models
- **Performance Metrics**: Track response times, token usage, and costs
- **Flexible Configuration**: Support for multiple providers through environment variables
- **Custom Question Sets**: Create and manage your own benchmark questions
- **Results Visualization**: Built-in viewer for analyzing benchmark results
- **Code-Only Prompting**: Specialized testing for code generation capabilities

**Quick Start**:
```bash
cd llm-benchmark
pip install -r requirements.txt
cp .env.example .env
# Configure your API keys in .env
python verify_setup.py
python run.py
```

**Documentation**:
- [Full Documentation](llm-benchmark/README.md) - Complete guide to the benchmark tool
- [Quick Start Guide](llm-benchmark/QUICKSTART.md) - Get up and running in 5 minutes
- [Questions Guide](llm-benchmark/QUESTIONS_GUIDE.md) - How to create custom benchmark questions
- [Code Prompting Guide](llm-benchmark/CODE_ONLY_PROMPTING.md) - Testing code generation
- [Changelog](llm-benchmark/CHANGELOG.md) - Version history and updates

## Getting Started

Each tool in this repository is self-contained with its own documentation and setup instructions. Navigate to the specific project directory for detailed guides.

### Prerequisites

- Python 3.8 or higher
- API keys for LLM providers you want to use
- Basic understanding of LLM APIs and concepts

### General Setup Pattern

Most tools follow a similar setup process:

1. **Navigate to the tool directory**
   ```bash
   cd <tool-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the tool**
   ```bash
   python run.py  # or the main script for that tool
   ```

## Related Projects

This repository is part of a larger ecosystem of LLM tools:

- **[LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy)** - Universal LLM API proxy with resilience and key management
- **[Mirrobot-agent](https://github.com/Mirrowel/Mirrobot-agent)** - AI-powered GitHub bot for automated issue analysis and PR reviews
- **[Mirrobot-py](https://github.com/Mirrowel/Mirrobot-py)** - Discord bot with LLM integration and advanced features

These projects work seamlessly together to provide a complete LLM development and deployment solution.

## Contributing

Contributions are welcome! Whether you want to:
- Add new tools to the collection
- Improve existing functionality
- Fix bugs or add features
- Enhance documentation

Please follow these guidelines:

1. **Fork the Repository** and clone it locally
2. **Create a Feature Branch** (`git checkout -b feature/amazing-feature`)
3. **Make Your Changes**, following existing code style and patterns
4. **Test Your Changes** thoroughly
5. **Commit Your Changes** with descriptive commit messages
6. **Push to Your Fork** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request** with a clear description of your changes

## Support

If you need help or have questions:

1. Check the documentation for the specific tool you're using
2. Visit the [GitHub Issues](https://github.com/Mirrowel/LLM-Tools/issues) page to report bugs or request features
3. Support the project on [Ko-fi](https://ko-fi.com/C0C0UZS4P) if you find it useful

## License

This project is open source. Please check individual tool directories for specific license information.

---

**Note**: This is an active project with tools being added and updated regularly. Star the repository to stay updated with new additions!
