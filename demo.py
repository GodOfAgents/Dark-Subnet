"""
Dark Subnet Demo - End-to-End Blind Inference Demonstration

This script demonstrates the full workflow of the Dark Subnet:
1. Train and compile an FHE model
2. Client encrypts sensitive data
3. Miner performs blind inference
4. Client decrypts result

This is designed for hackathon demonstration - showing judges
that miners never see the actual data.

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def check_model_exists() -> bool:
    """Check if FHE model is compiled."""
    model_path = Path(__file__).parent / "fhe_models" / "credit_scorer"
    return (model_path / "server.zip").exists() and (model_path / "client.zip").exists()


def train_model():
    """Train and compile the FHE model."""
    console.print("\n[bold yellow]Step 1: Training FHE Model[/bold yellow]")
    console.print("─" * 50)
    
    # Import and run training
    from fhe_models.train_model import (
        generate_credit_scoring_data,
        train_logistic_regression,
        compile_and_save_model,
        verify_fhe_execution,
    )
    from sklearn.model_selection import train_test_split
    
    # Generate data
    console.print("[cyan]Generating synthetic credit data...[/cyan]")
    X, y = generate_credit_scoring_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train
    model = train_logistic_regression(X_train, y_train, X_test, y_test, n_bits=8)
    
    # Compile
    output_dir = Path(__file__).parent / "fhe_models" / "credit_scorer"
    compile_and_save_model(model, X_train, output_dir)
    
    # Verify
    verify_fhe_execution(model, X_test, y_test)
    
    console.print("[green]✓ Model trained and compiled![/green]")
    return model


def simulate_blind_inference():
    """Simulate the full blind inference workflow."""
    console.print("\n[bold yellow]Step 2: Simulating Blind Inference[/bold yellow]")
    console.print("─" * 50)
    
    from concrete.ml.deployment import FHEModelClient, FHEModelServer
    
    model_path = Path(__file__).parent / "fhe_models" / "credit_scorer"
    
    # === CLIENT SIDE ===
    console.print("\n[bold cyan]👤 CLIENT SIDE (Hospital/Bank)[/bold cyan]")
    
    # Load client
    client = FHEModelClient(str(model_path))
    evaluation_keys = client.get_serialized_evaluation_keys()
    console.print(f"[green]✓ Loaded FHE client[/green]")
    console.print(f"[green]✓ Evaluation keys: {len(evaluation_keys):,} bytes[/green]")
    
    # Create test data (this is the SENSITIVE data)
    console.print("\n[yellow]Creating sensitive patient data...[/yellow]")
    sensitive_data = np.array([[
        0.65,   # Age: 65 (normalized)
        0.4,    # Income: Average
        0.7,    # Debt ratio: High
        0.3,    # Few accounts
        0.3,    # Poor payment history
        0.8,    # High credit utilization
        0.2,    # Short credit history
        0.6,    # Several inquiries
        0.0,    # No mortgage
        1.0,    # Has defaults!
    ]], dtype=np.float32)
    
    # Show what the data looks like
    feature_table = Table(title="🔓 Original Patient Data (SENSITIVE)")
    feature_table.add_column("Feature", style="cyan")
    feature_table.add_column("Value", style="yellow")
    
    feature_names = [
        "Age", "Income", "Debt Ratio", "Accounts",
        "Payment History", "Credit Util", "Credit Years",
        "Inquiries", "Has Mortgage", "Has Defaults"
    ]
    
    for name, value in zip(feature_names, sensitive_data[0]):
        feature_table.add_row(name, f"{value:.2f}")
    
    console.print(feature_table)
    
    # Encrypt the data
    console.print("\n[yellow]Encrypting data with FHE...[/yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Encrypting...", total=None)
        encrypted_data = client.quantize_encrypt_serialize(sensitive_data)
        progress.remove_task(task)
    
    console.print(f"[green]✓ Encrypted data: {len(encrypted_data):,} bytes[/green]")
    
    # Show a snippet of the ciphertext
    console.print("\n[dim]Ciphertext (first 100 bytes):[/dim]")
    console.print(f"[dim]{encrypted_data[:100].hex()}...[/dim]")
    
    # === MINER SIDE ===
    console.print("\n" + "=" * 50)
    console.print("\n[bold magenta]⚡ MINER SIDE (Blind Computation)[/bold magenta]")
    
    # Load server (miner only has this)
    server = FHEModelServer(str(model_path))
    console.print("[green]✓ Loaded FHE server[/green]")
    
    # Show what the miner sees
    console.print("\n[bold red]🔒 What the MINER sees:[/bold red]")
    console.print(Panel(
        f"[dim]{encrypted_data[:200].hex()}...[/dim]\n\n"
        "[bold yellow]The miner sees ONLY mathematical noise.[/bold yellow]\n"
        "[yellow]It cannot determine:[/yellow]\n"
        "  • Patient age\n"
        "  • Income level\n"
        "  • Default history\n"
        "  • ANY personal information[/yellow]",
        title="🔒 Miner's View (Ciphertext Only)",
        border_style="red",
    ))
    
    # Perform blind inference
    console.print("\n[yellow]Performing BLIND inference on encrypted data...[/yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing on ciphertext...", total=None)
        start_time = time.time()
        encrypted_result = server.run(encrypted_data, evaluation_keys)
        inference_time = time.time() - start_time
        progress.remove_task(task)
    
    console.print(f"[green]✓ Blind inference complete: {inference_time*1000:.2f}ms[/green]")
    console.print(f"[green]✓ Encrypted result: {len(encrypted_result):,} bytes[/green]")
    
    # === BACK TO CLIENT ===
    console.print("\n" + "=" * 50)
    console.print("\n[bold cyan]👤 BACK TO CLIENT (Decryption)[/bold cyan]")
    
    # Decrypt the result
    console.print("[yellow]Decrypting result...[/yellow]")
    result = client.deserialize_decrypt_dequantize(encrypted_result)
    prediction = int(result[0])
    
    # Interpret result
    if prediction == 1:
        risk_level = "[bold red]HIGH RISK[/bold red]"
        recommendation = "Recommend specialist follow-up"
    else:
        risk_level = "[bold green]LOW RISK[/bold green]"
        recommendation = "Continue routine monitoring"
    
    result_panel = Panel(
        f"Prediction: {risk_level}\n"
        f"Recommendation: {recommendation}\n\n"
        f"[dim]Inference time: {inference_time*1000:.2f}ms[/dim]",
        title="✅ Decrypted Result",
        border_style="green",
    )
    console.print(result_panel)
    
    return True


def demonstrate_honey_pot():
    """Demonstrate the honey pot verification mechanism."""
    console.print("\n[bold yellow]Step 3: Demonstrating Honey Pot Verification[/bold yellow]")
    console.print("─" * 50)
    
    from concrete.ml.deployment import FHEModelClient, FHEModelServer
    
    model_path = Path(__file__).parent / "fhe_models" / "credit_scorer"
    
    client = FHEModelClient(str(model_path))
    server = FHEModelServer(str(model_path))
    evaluation_keys = client.get_serialized_evaluation_keys()
    
    # Create honey pot traps
    console.print("\n[bold cyan]🍯 Creating Honey Pot Traps[/bold cyan]")
    
    traps = [
        {
            "name": "High Risk Trap",
            "features": np.array([[0.9, 0.1, 0.9, 0.8, 0.1, 0.9, 0.1, 0.8, 0.0, 1.0]], dtype=np.float32),
            "expected": 1,
        },
        {
            "name": "Low Risk Trap",
            "features": np.array([[0.3, 0.9, 0.1, 0.2, 0.9, 0.1, 0.9, 0.1, 1.0, 0.0]], dtype=np.float32),
            "expected": 0,
        },
    ]
    
    results_table = Table(title="🍯 Honey Pot Verification Results")
    results_table.add_column("Trap", style="cyan")
    results_table.add_column("Expected", style="yellow")
    results_table.add_column("Actual", style="yellow")
    results_table.add_column("Status", style="green")
    
    for trap in traps:
        # Encrypt trap
        encrypted = client.quantize_encrypt_serialize(trap["features"])
        
        # Miner processes (doesn't know it's a trap)
        encrypted_result = server.run(encrypted, evaluation_keys)
        
        # Validator decrypts (only trap data)
        result = client.deserialize_decrypt_dequantize(encrypted_result)
        actual = int(result[0])
        
        passed = actual == trap["expected"]
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        
        results_table.add_row(
            trap["name"],
            str(trap["expected"]),
            str(actual),
            status,
        )
    
    console.print(results_table)
    
    console.print(Panel(
        "[bold]The Trust Sandwich Protocol:[/bold]\n\n"
        "1. Validator creates traps with KNOWN expected outputs\n"
        "2. Traps are mixed with real client requests\n"
        "3. Miner processes ALL requests (can't distinguish traps)\n"
        "4. Validator decrypts ONLY trap results\n"
        "5. If miner passes traps → statistically trust real results\n\n"
        "[yellow]Result: Blind verification without seeing actual data![/yellow]",
        title="How Honey Pot Verification Works",
        border_style="cyan",
    ))


def print_summary():
    """Print final summary for judges."""
    console.print("\n" + "=" * 60)
    console.print("\n[bold green]📊 SUMMARY FOR HACKATHON JUDGES[/bold green]\n")
    
    summary_table = Table(title="Dark Subnet vs Standard Subnet")
    summary_table.add_column("Feature", style="cyan")
    summary_table.add_column("Standard Subnet", style="yellow")
    summary_table.add_column("Dark Subnet", style="green")
    
    comparisons = [
        ("Data Visibility", "Public", "🔒 ZERO"),
        ("Miner Sees", "Raw text/images", "🔒 Mathematical noise"),
        ("Verification", "Redundant (2 miners)", "🍯 Honey Pots"),
        ("Hardware", "GPU Required", "✅ CPU/GPU Agnostic"),
        ("Use Cases", "Chatbots, Image Gen", "🏥 Medical, 💰 Financial"),
        ("Privacy", "None", "✅ Full FHE"),
    ]
    
    for feature, standard, dark in comparisons:
        summary_table.add_row(feature, standard, dark)
    
    console.print(summary_table)
    
    console.print(Panel(
        "[bold]What We Demonstrated:[/bold]\n\n"
        "✅ [green]Blind Inference[/green]: Miners compute on encrypted data\n"
        "✅ [green]Blind Verification[/green]: Validators verify via honey pots\n"
        "✅ [green]Client Privacy[/green]: Only client decrypts results\n"
        "✅ [green]Fast Execution[/green]: LogisticRegression in <1 second\n\n"
        "[bold cyan]Technology Stack:[/bold cyan]\n"
        "• Zama Concrete ML for FHE\n"
        "• Bittensor for decentralized subnet\n"
        "• Python for cross-platform support",
        title="🏆 Hackathon Demo Complete",
        border_style="green",
    ))


def main():
    """Run the complete demo."""
    console.print("\n" + "=" * 60)
    console.print(Panel(
        "[bold white]Dark Subnet[/bold white]\n"
        "[dim]Privacy-First Bittensor Subnet with FHE[/dim]\n\n"
        "[cyan]Demonstrating:[/cyan]\n"
        "• Blind Inference (miners see noise)\n"
        "• Blind Verification (honey pot traps)\n"
        "• Client Oracle (external proof)",
        title="🌑 HACKATHON DEMO",
        border_style="magenta",
    ))
    
    try:
        # Check if model exists
        if not check_model_exists():
            console.print("[yellow]FHE model not found. Training now...[/yellow]")
            train_model()
        else:
            console.print("[green]✓ FHE model already compiled[/green]")
        
        # Run blind inference demo
        simulate_blind_inference()
        
        # Demonstrate honey pot verification
        demonstrate_honey_pot()
        
        # Print summary
        print_summary()
        
    except ImportError as e:
        console.print(f"\n[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install -r requirements.txt[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
