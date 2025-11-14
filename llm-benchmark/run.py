#!/usr/bin/env python3
"""
Main entry point for the LLM Benchmark system.
Reads configuration from config.yaml and runs benchmarks.

Usage:
    python run.py                    # Run with config.yaml settings
    python run.py --config custom.yaml  # Use custom config file
    python run.py --help             # Show help
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.rotator_library.client import RotatingClient
from src.runner import BenchmarkRunner
from src.config_loader import ConfigLoader

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark System - Evaluate multiple LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Edit config.yaml to set models, categories, and other settings.
  All benchmark parameters are read from the config file.

Examples:
  python run.py                       # Run with default config.yaml
  python run.py --config custom.yaml  # Use custom config file
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()

    # Load configuration
    try:
        config = ConfigLoader(args.config)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nCreate a config.yaml file to get started.")
        print("See config.yaml for an example configuration.\n")
        return
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}\n")
        return

    print("\n" + "="*60)
    print("LLM Benchmark System")
    print("="*60)
    print(f"\nConfiguration: {args.config}")
    print(f"Models: {', '.join(config.models)}")
    print(f"Judge: {config.judge_model}")
    if config.categories:
        print(f"Categories: {', '.join(config.categories)}")
    if config.question_ids:
        print(f"Questions: {', '.join(config.question_ids)}")
    print("")

    # Collect API keys from environment
    api_keys = defaultdict(list)
    for key, value in os.environ.items():
        if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
            parts = key.split("_API_KEY")
            provider = parts[0].lower()
            if provider not in api_keys:
                api_keys[provider] = []
            api_keys[provider].append(value)

    if not api_keys:
        print("❌ ERROR: No provider API keys found in environment variables.")
        print("Please set API keys in your .env file (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        print("\nExample .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("")
        return

    # Create client with retry settings from config
    client = RotatingClient(
        api_keys=dict(api_keys),
        max_retries=config.max_retries_per_key,
        global_timeout=config.global_timeout
    )

    try:
        # Create and run benchmark
        runner = BenchmarkRunner(
            client=client,
            judge_model=config.judge_model,
            questions_dir=config.questions_dir,
            results_dir=config.results_dir,
            model_system_instructions=config.all_model_system_instructions,
            model_options=config.all_model_options,
            code_formatting_enabled=config.code_formatting_enabled,
            code_formatting_instruction=config.code_formatting_instruction
        )

        run_id = await runner.run_benchmark(
            models=config.models,
            categories=config.categories,
            question_ids=config.question_ids,
            max_concurrent=config.max_concurrent,
            provider_concurrency=config.provider_concurrency
        )

        print(f"\n✅ Benchmark complete! Run ID: {run_id}")
        print(f"📊 View results: python viewer/server.py")
        print(f"📁 Results directory: {config.results_dir}/{run_id}\n")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
