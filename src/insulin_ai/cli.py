#!/usr/bin/env python3
"""
Command-Line Interface for Insulin-AI

Provides command-line tools for running various components of the insulin-ai package.
"""

import click
import sys
import os
from pathlib import Path
from typing import Optional
import logging

from . import get_version, get_package_info


@click.group()
@click.version_option(version=get_version(), prog_name="insulin-ai")
@click.option(
    "--verbose", "-v", 
    count=True, 
    help="Increase verbosity (use -v, -vv, or -vvv)"
)
@click.option(
    "--config-dir", 
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Configuration directory path"
)
def main(verbose: int, config_dir: Optional[str]):
    """Insulin-AI: AI-Powered Drug Delivery System"""
    # Set up logging based on verbosity
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set config directory if provided
    if config_dir:
        os.environ["INSULIN_AI_CONFIG_DIR"] = config_dir


@main.command()
def info():
    """Display package information"""
    info_data = get_package_info()
    click.echo(f"Insulin-AI Package Information:")
    click.echo(f"  Version: {info_data['version']}")
    click.echo(f"  Author: {info_data['author']}")
    click.echo(f"  License: {info_data['license']}")
    click.echo(f"  Package Root: {info_data['package_root']}")
    click.echo(f"  MD Available: {info_data['md_available']}")
    click.echo(f"  Automation Available: {info_data['automation_available']}")


@main.command()
@click.option(
    "--host", "-h", 
    default="localhost", 
    help="Host to bind the web interface to"
)
@click.option(
    "--port", "-p", 
    default=8501, 
    type=int, 
    help="Port to bind the web interface to"
)
@click.option(
    "--openai-key", 
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (can also be set via OPENAI_API_KEY env var)"
)
def web(host: str, port: int, openai_key: Optional[str]):
    """Launch the Streamlit web interface"""
    try:
        import streamlit.web.cli as streamlit_cli
    except ImportError:
        click.echo("Error: Streamlit is not installed. Install with 'pip install streamlit'")
        sys.exit(1)
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    # Path to the main app file
    app_file = Path(__file__).parent / "app.py"
    
    if not app_file.exists():
        click.echo(f"Error: App file not found at {app_file}")
        sys.exit(1)
    
    # Launch streamlit
    sys.argv = [
        "streamlit", "run", str(app_file),
        "--server.address", host,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ]
    
    streamlit_cli.main()


@main.command()
@click.option(
    "--material-request", "-m",
    required=True,
    help="Material specification request for PSMILES generation"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False),
    help="Output file for generated PSMILES (JSON format)"
)
@click.option(
    "--model", 
    default="gpt-3.5-turbo",
    help="OpenAI model to use for generation"
)
@click.option(
    "--num-candidates", "-n",
    default=5,
    type=int,
    help="Number of candidate PSMILES to generate"
)
def generate_psmiles(material_request: str, output: Optional[str], model: str, num_candidates: int):
    """Generate PSMILES for a given material request"""
    try:
        from .core.psmiles_generator import PSMILESGenerator
    except ImportError as e:
        click.echo(f"Error importing PSMILESGenerator: {e}")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    click.echo(f"Generating PSMILES for: {material_request}")
    click.echo(f"Using model: {model}")
    click.echo(f"Number of candidates: {num_candidates}")
    
    # Initialize generator
    generator = PSMILESGenerator(
        model_type='openai',
        openai_model=model,
        temperature=0.7
    )
    
    # Generate PSMILES
    try:
        results = generator.generate_truly_diverse_candidates(
            base_request=material_request,
            num_candidates=num_candidates,
            enable_functionalization=True,
            diversity_threshold=0.4
        )
        
        if results.get('success'):
            click.echo(f"✅ Generation successful!")
            click.echo(f"Best candidate: {results.get('best_candidate')}")
            
            if output:
                import json
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                click.echo(f"Results saved to: {output}")
            else:
                click.echo("All candidates:")
                for i, candidate in enumerate(results.get('candidates', []), 1):
                    click.echo(f"  {i}. {candidate}")
        else:
            click.echo(f"❌ Generation failed: {results.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error during generation: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--query", "-q",
    required=True,
    help="Literature search query"
)
@click.option(
    "--max-papers", "-m",
    default=10,
    type=int,
    help="Maximum number of papers to retrieve"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False),
    help="Output file for literature results (JSON format)"
)
def mine_literature(query: str, max_papers: int, output: Optional[str]):
    """Mine literature for relevant papers and insights"""
    try:
        from .core.literature_mining_system import MaterialsLiteratureMiner
    except ImportError as e:
        click.echo(f"Error importing MaterialsLiteratureMiner: {e}")
        sys.exit(1)
    
    click.echo(f"Mining literature for: {query}")
    click.echo(f"Max papers: {max_papers}")
    
    # Initialize miner
    miner = MaterialsLiteratureMiner(
        model_type="openai",
        openai_model="gpt-3.5-turbo"
    )
    
    try:
        results = miner.search_and_analyze_papers(
            query=query,
            max_papers=max_papers
        )
        
        if results.get('success'):
            papers = results.get('papers', [])
            click.echo(f"✅ Found {len(papers)} relevant papers")
            
            if output:
                import json
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                click.echo(f"Results saved to: {output}")
            else:
                for i, paper in enumerate(papers[:5], 1):  # Show first 5
                    click.echo(f"  {i}. {paper.get('title', 'No title')}")
                    click.echo(f"     Authors: {', '.join(paper.get('authors', []))}")
                    click.echo(f"     Year: {paper.get('year', 'Unknown')}")
                    click.echo()
        else:
            click.echo(f"❌ Literature mining failed: {results.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error during literature mining: {e}")
        sys.exit(1)


@main.command()
def test_installation():
    """Test the installation and core functionality"""
    click.echo("Testing Insulin-AI installation...")
    
    # Test core imports
    tests = [
        ("Core package", lambda: __import__("insulin_ai")),
        ("PSMILESGenerator", lambda: __import__("insulin_ai.core.psmiles_generator")),
        ("PSMILESProcessor", lambda: __import__("insulin_ai.core.psmiles_processor")),
        ("Literature Miner", lambda: __import__("insulin_ai.core.literature_mining_system")),
        ("Chatbot System", lambda: __import__("insulin_ai.core.chatbot_system")),
    ]
    
    optional_tests = [
        ("MD Integration", lambda: __import__("insulin_ai.integration.analysis.simple_md_integration")),
        ("Simulation Automation", lambda: __import__("insulin_ai.integration.automation.simulation_automation")),
    ]
    
    # Run core tests
    all_passed = True
    for test_name, test_func in tests:
        try:
            test_func()
            click.echo(f"✅ {test_name}: OK")
        except Exception as e:
            click.echo(f"❌ {test_name}: FAILED ({e})")
            all_passed = False
    
    # Run optional tests
    click.echo("\nOptional components:")
    for test_name, test_func in optional_tests:
        try:
            test_func()
            click.echo(f"✅ {test_name}: OK")
        except Exception as e:
            click.echo(f"⚠️  {test_name}: Not available ({e})")
    
    if all_passed:
        click.echo("\n🎉 Installation test passed! All core components are working.")
    else:
        click.echo("\n❌ Installation test failed. Some core components are missing or broken.")
        sys.exit(1)


if __name__ == "__main__":
    main() 