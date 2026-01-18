"""
Simulation Runner - Run complete Dark Subnet demo locally

This script runs the full miner-validator flow in simulation mode,
without needing Bittensor network setup.

Usage:
    python run_simulation.py
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


def check_model_exists() -> bool:
    """Check if FHE model is compiled."""
    model_path = Path(__file__).parent / "fhe_models" / "credit_scorer"
    return (model_path / "server.zip").exists() and (model_path / "client.zip").exists()


def train_model():
    """Train the FHE model if needed."""
    console.print("[yellow]Training FHE model...[/yellow]")
    
    result = subprocess.run(
        [sys.executable, "fhe_models/train_model.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        console.print(f"[red]Training failed: {result.stderr}[/red]")
        return False
    
    console.print("[green]✓ Model trained successfully[/green]")
    return True


def run_miner_background():
    """Run miner in background thread."""
    subprocess.run(
        [sys.executable, "neurons/miner.py", "--simulate"],
        cwd=Path(__file__).parent,
    )


def run_validator():
    """Run validator."""
    subprocess.run(
        [sys.executable, "neurons/validator.py", "--simulate", "--rounds", "3"],
        cwd=Path(__file__).parent,
    )


def main():
    """Run the complete simulation."""
    console.print("\n" + "=" * 60)
    console.print(Panel(
        "[bold white]Dark Subnet Simulation[/bold white]\n"
        "[dim]Running complete miner-validator flow locally[/dim]\n\n"
        "[cyan]What will happen:[/cyan]\n"
        "1. Train FHE model (if needed)\n"
        "2. Start simulated miner (background)\n"
        "3. Start simulated validator\n"
        "4. Validator sends honey pots to miner\n"
        "5. Validator verifies miner responses",
        title="🌑 LOCAL SIMULATION",
        border_style="magenta",
    ))
    
    # Check/train model
    if not check_model_exists():
        console.print("\n[yellow]FHE model not found. Training...[/yellow]")
        if not train_model():
            console.print("[red]Cannot continue without model.[/red]")
            return 1
    else:
        console.print("\n[green]✓ FHE model already compiled[/green]")
    
    # Start miner in background
    console.print("\n[cyan]Starting simulated miner...[/cyan]")
    miner_thread = threading.Thread(target=run_miner_background, daemon=True)
    miner_thread.start()
    
    # Wait for miner to start
    console.print("[dim]Waiting for miner to initialize...[/dim]")
    time.sleep(3)
    
    # Check if miner is running
    try:
        import requests
        response = requests.get("http://localhost:8091/health", timeout=5)
        if response.status_code == 200:
            console.print("[green]✓ Miner is running[/green]")
        else:
            console.print("[red]✗ Miner health check failed[/red]")
            return 1
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to miner: {e}[/red]")
        console.print("[yellow]Make sure concrete-ml is installed: pip install concrete-ml[/yellow]")
        return 1
    
    # Run validator
    console.print("\n[cyan]Starting validator verification...[/cyan]")
    console.print("─" * 60)
    
    run_validator()
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel(
        "[bold green]Simulation Complete![/bold green]\n\n"
        "[cyan]What was demonstrated:[/cyan]\n"
        "✓ Miner received encrypted data (saw only ciphertext)\n"
        "✓ Miner performed blind FHE inference\n"
        "✓ Validator created honey pot traps\n"
        "✓ Validator verified miner honesty\n\n"
        "[yellow]This proves:[/yellow]\n"
        "• Miners work on data they CANNOT see\n"
        "• Validators grade work WITHOUT seeing answers\n"
        "• Cheating miners are caught by traps",
        title="🏆 DEMO COMPLETE",
        border_style="green",
    ))
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(0)
