"""
FHE Model Training Script - Credit Score Classifier

This script trains a LogisticRegression model on synthetic credit scoring data,
then compiles it to an FHE circuit for privacy-preserving inference.

The compiled model can run inference on encrypted data - the server (miner)
never sees the actual credit data (age, income, debt, etc.)

Usage:
    python train_model.py [--model logistic|xgboost] [--output-dir PATH]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

console = Console()


def generate_credit_scoring_data(n_samples: int = 1000, random_state: int = 42):
    """
    Generate synthetic credit scoring data.
    
    Features represent:
    - age (normalized)
    - annual_income (normalized)
    - debt_to_income_ratio
    - num_credit_accounts
    - payment_history_score
    - credit_utilization
    - years_of_credit_history
    - num_hard_inquiries
    - has_mortgage (binary)
    - has_default_history (binary)
    
    Target: 0 = Low Risk, 1 = High Risk
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=random_state,
    )
    
    # Normalize features to [0, 1] range for better FHE performance
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    return X, y


def train_logistic_regression(X_train, y_train, X_test, y_test, n_bits: int = 8):
    """Train a quantized Logistic Regression model for FHE."""
    from concrete.ml.sklearn import LogisticRegression
    
    console.print("[cyan]Training LogisticRegression model...[/cyan]")
    
    model = LogisticRegression(n_bits=n_bits)
    model.fit(X_train, y_train)
    
    # Evaluate in clear
    y_pred_clear = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_clear)
    
    console.print(f"[green]Clear accuracy: {accuracy:.4f}[/green]")
    
    return model


def train_xgboost(X_train, y_train, X_test, y_test, n_bits: int = 8):
    """Train a quantized XGBoost model for FHE."""
    from concrete.ml.sklearn import XGBClassifier
    
    console.print("[cyan]Training XGBoost model...[/cyan]")
    
    model = XGBClassifier(
        n_bits=n_bits,
        n_estimators=10,  # Keep small for fast FHE
        max_depth=4,
    )
    model.fit(X_train, y_train)
    
    # Evaluate in clear
    y_pred_clear = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_clear)
    
    console.print(f"[green]Clear accuracy: {accuracy:.4f}[/green]")
    
    return model


def compile_and_save_model(model, X_calibration, output_dir: Path):
    """Compile model to FHE circuit and save deployment artifacts."""
    from concrete.ml.deployment import FHEModelDev
    
    console.print("[cyan]Compiling to FHE circuit...[/cyan]")
    
    # Compile the model (this creates the FHE circuit)
    model.compile(X_calibration)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save using FHEModelDev for client/server deployment
    console.print(f"[cyan]Saving to {output_dir}...[/cyan]")
    
    fhe_dev = FHEModelDev(model)
    fhe_dev.save(output_dir)
    
    console.print("[green]✓ Saved client.zip (for encryption/decryption)[/green]")
    console.print("[green]✓ Saved server.zip (for blind inference)[/green]")
    
    return output_dir


def verify_fhe_execution(model, X_test, y_test, n_samples: int = 5):
    """Verify FHE execution matches clear execution."""
    console.print("\n[cyan]Verifying FHE execution...[/cyan]")
    
    # Take a few samples for verification
    X_sample = X_test[:n_samples]
    y_true = y_test[:n_samples]
    
    # Clear prediction
    y_pred_clear = model.predict(X_sample)
    
    # FHE prediction (simulated for speed during training)
    y_pred_fhe = model.predict(X_sample, fhe="simulate")
    
    # Compare
    match = (y_pred_clear == y_pred_fhe).all()
    
    if match:
        console.print("[green]✓ FHE predictions match clear predictions![/green]")
    else:
        console.print("[yellow]⚠ FHE predictions differ from clear (expected with quantization)[/yellow]")
    
    console.print(f"  Clear predictions:  {y_pred_clear}")
    console.print(f"  FHE predictions:    {y_pred_fhe}")
    console.print(f"  True labels:        {y_true}")
    
    return match


def main():
    parser = argparse.ArgumentParser(description="Train FHE Credit Scoring Model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Model type to train (default: logistic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "credit_scorer",
        help="Output directory for compiled model",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=8,
        help="Quantization bits (lower = faster FHE, default: 8)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)",
    )
    
    args = parser.parse_args()
    
    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Dark Subnet - FHE Model Training       ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
    
    # Generate data
    console.print(f"[cyan]Generating {args.n_samples} credit scoring samples...[/cyan]")
    X, y = generate_credit_scoring_data(n_samples=args.n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    console.print(f"[green]Train: {len(X_train)} samples, Test: {len(X_test)} samples[/green]")
    
    # Train model
    if args.model == "logistic":
        model = train_logistic_regression(X_train, y_train, X_test, y_test, args.n_bits)
    else:
        model = train_xgboost(X_train, y_train, X_test, y_test, args.n_bits)
    
    # Compile to FHE
    compile_and_save_model(model, X_train, args.output_dir)
    
    # Verify FHE execution
    verify_fhe_execution(model, X_test, y_test)
    
    # Print summary
    console.print("\n[bold green]═══ Training Complete ═══[/bold green]")
    console.print(f"  Model: {args.model}")
    console.print(f"  Bits: {args.n_bits}")
    console.print(f"  Output: {args.output_dir}")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Run miner: python neurons/miner.py")
    console.print("  2. Run validator: python neurons/validator.py")
    console.print("  3. Run demo: python demo.py")


if __name__ == "__main__":
    main()
