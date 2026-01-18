"""
Blind Validator - Honey Pot Verification System

This validator grades miner performance WITHOUT being able to decrypt
the actual inference results. It uses a "Trust Sandwich" mechanism:

1. Create HONEY POT traps with known expected outputs
2. Mix traps with real client requests
3. Verify ONLY the trap results (validator can decrypt these)
4. Statistically trust that real results are computed correctly

Usage:
    # Simulation mode (no Bittensor required)
    python validator.py --simulate
    
    # Live Bittensor mode
    python validator.py --netuid 1 --wallet.name validator --wallet.hotkey default
"""

import argparse
import asyncio
import random
import time
import uuid
import json
import requests
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class TrapResult:
    """Result of a honey pot verification."""
    def __init__(self, miner_id: str, passed: bool, expected: int, actual: int, latency_ms: float):
        self.miner_id = miner_id
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.latency_ms = latency_ms


# ============================================================
# SIMULATION MODE - No Bittensor Required
# ============================================================

class SimulatedValidator:
    """
    A simulated validator that runs without Bittensor network.
    
    Demonstrates honey pot verification against local miners.
    """
    
    def __init__(self, miner_urls: list[str] = None):
        self.miner_urls = miner_urls or ["http://localhost:8091"]
        self.fhe_client = None
        self.evaluation_keys = None
        self.scores = {}  # miner_url -> score
        
        self._load_fhe_client()
    
    def _load_fhe_client(self):
        """Load FHE client for creating and verifying traps."""
        try:
            from concrete.ml.deployment import FHEModelClient
            
            model_path = Path(__file__).parent.parent / "fhe_models" / "credit_scorer"
            
            if (model_path / "client.zip").exists():
                self.fhe_client = FHEModelClient(str(model_path))
                self.evaluation_keys = self.fhe_client.get_serialized_evaluation_keys()
                console.print("[green]✓ Loaded FHE client for trap verification[/green]")
            else:
                console.print("[yellow]⚠ Model not compiled. Run: python fhe_models/train_model.py[/yellow]")
        except ImportError:
            console.print("[red]✗ Concrete ML not installed. Run: pip install concrete-ml[/red]")
    
    def create_honey_pot(self) -> tuple[bytes, int, str]:
        """
        Create a honey pot trap with KNOWN expected output.
        
        Returns:
            Tuple of (encrypted_trap, expected_output, description)
        """
        if self.fhe_client is None:
            raise RuntimeError("FHE client not loaded")
        
        # Trap profiles with known outputs
        trap_profiles = [
            {
                "features": [0.95, 0.1, 0.9, 0.8, 0.1, 0.95, 0.1, 0.9, 0.0, 1.0],
                "expected": 1,
                "description": "🔴 High Risk: Elderly, low income, high debt, defaults"
            },
            {
                "features": [0.85, 0.15, 0.85, 0.75, 0.15, 0.9, 0.1, 0.85, 0.0, 1.0],
                "expected": 1,
                "description": "🔴 High Risk: Poor payment history, many inquiries"
            },
            {
                "features": [0.3, 0.9, 0.1, 0.2, 0.9, 0.15, 0.85, 0.05, 1.0, 0.0],
                "expected": 0,
                "description": "🟢 Low Risk: Prime borrower, excellent history"
            },
            {
                "features": [0.4, 0.85, 0.15, 0.25, 0.85, 0.2, 0.8, 0.1, 1.0, 0.0],
                "expected": 0,
                "description": "🟢 Low Risk: Stable income, mortgage holder"
            },
        ]
        
        trap = random.choice(trap_profiles)
        features = np.array([trap["features"]], dtype=np.float32)
        
        # Encrypt the trap
        encrypted_trap = self.fhe_client.quantize_encrypt_serialize(features)
        
        return encrypted_trap, trap["expected"], trap["description"]
    
    def verify_trap_result(self, encrypted_result: bytes, expected: int) -> tuple[bool, int]:
        """Verify a honey pot result by decrypting it."""
        if self.fhe_client is None:
            return False, -1
        
        try:
            result = self.fhe_client.deserialize_decrypt_dequantize(encrypted_result)
            actual = int(result[0])
            passed = (actual == expected)
            return passed, actual
        except Exception as e:
            console.print(f"[red]Decryption failed: {e}[/red]")
            return False, -1
    
    def query_miner(self, miner_url: str, encrypted_data: bytes) -> dict:
        """Query a simulated miner via HTTP."""
        try:
            response = requests.post(
                f"{miner_url}/inference",
                json={
                    "encrypted_data": encrypted_data.hex(),
                    "evaluation_keys": self.evaluation_keys.hex(),
                },
                timeout=60,
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("encrypted_prediction"):
                    data["encrypted_prediction"] = bytes.fromhex(data["encrypted_prediction"])
                return data
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection refused - is miner running?"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def validate_miners(self) -> list[TrapResult]:
        """Run validation against all configured miners."""
        results = []
        
        console.print("\n[bold cyan]🍯 HONEY POT VERIFICATION[/bold cyan]")
        console.print("─" * 50)
        
        for miner_url in self.miner_urls:
            console.print(f"\n[cyan]Testing miner: {miner_url}[/cyan]")
            
            # Create honey pot
            encrypted_trap, expected, description = self.create_honey_pot()
            console.print(f"[dim]Trap: {description}[/dim]")
            console.print(f"[dim]Expected output: {expected}[/dim]")
            
            # Query miner
            start_time = time.time()
            response = self.query_miner(miner_url, encrypted_trap)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.get("success") and response.get("encrypted_prediction"):
                # Verify the trap
                passed, actual = self.verify_trap_result(
                    response["encrypted_prediction"],
                    expected
                )
                
                result = TrapResult(
                    miner_id=miner_url,
                    passed=passed,
                    expected=expected,
                    actual=actual,
                    latency_ms=latency_ms,
                )
                results.append(result)
                
                # Update scores
                if miner_url not in self.scores:
                    self.scores[miner_url] = 0.5
                
                if passed:
                    self.scores[miner_url] = 0.9 * self.scores[miner_url] + 0.1 * 1.0
                    console.print(f"[bold green]✓ PASSED[/bold green] - Miner computed correctly!")
                else:
                    self.scores[miner_url] = 0.9 * self.scores[miner_url] + 0.1 * 0.0
                    console.print(f"[bold red]✗ FAILED[/bold red] - Expected {expected}, got {actual}")
                    console.print("[red]  → Miner may be cheating or broken![/red]")
            else:
                error = response.get("error", "Unknown error")
                console.print(f"[red]✗ ERROR: {error}[/red]")
                results.append(TrapResult(miner_url, False, expected, -1, latency_ms))
        
        return results
    
    def print_results(self, results: list[TrapResult]):
        """Print verification results table."""
        table = Table(title="🔒 Blind Verification Results")
        table.add_column("Miner", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Expected", style="yellow")
        table.add_column("Actual", style="yellow")
        table.add_column("Latency", style="magenta")
        table.add_column("Score", style="blue")
        
        for result in results:
            status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
            score = self.scores.get(result.miner_id, 0)
            
            table.add_row(
                result.miner_id.replace("http://", ""),
                status,
                str(result.expected),
                str(result.actual) if result.actual >= 0 else "N/A",
                f"{result.latency_ms:.0f}ms",
                f"{score:.2f}",
            )
        
        console.print("\n")
        console.print(table)
    
    def print_verification_explanation(self):
        """Print explanation of how verification works."""
        console.print(Panel(
            "[bold]How Blind Verification Works:[/bold]\n\n"
            "1. Validator creates a [cyan]TRAP[/cyan] with known expected output\n"
            "   Example: High-risk profile → Must return 1\n\n"
            "2. Trap is [cyan]ENCRYPTED[/cyan] with FHE (looks like noise)\n"
            "   Miner cannot tell it's a trap\n\n"
            "3. Miner processes the [cyan]CIPHERTEXT[/cyan]\n"
            "   Returns encrypted result\n\n"
            "4. Validator [cyan]DECRYPTS[/cyan] only the trap result\n"
            "   Compares with expected output\n\n"
            "5. If correct → [green]Miner is HONEST[/green]\n"
            "   If wrong → [red]Miner is CHEATING[/red]\n\n"
            "[yellow]Key Insight: Miner cannot distinguish traps from real data![/yellow]",
            title="🍯 The Trust Sandwich Protocol",
            border_style="cyan",
        ))
    
    def run(self, rounds: int = 5, interval: float = 5.0):
        """Run continuous validation."""
        console.print("\n[bold green]═══ Simulated Validator Running ═══[/bold green]")
        console.print(f"  Mode: [cyan]SIMULATION[/cyan] (no Bittensor)")
        console.print(f"  Miners: [cyan]{', '.join(self.miner_urls)}[/cyan]")
        console.print(f"  Rounds: [cyan]{rounds}[/cyan]")
        console.print(f"  Interval: [cyan]{interval}s[/cyan]")
        
        self.print_verification_explanation()
        
        for round_num in range(1, rounds + 1):
            console.print(f"\n[bold yellow]═══ Validation Round {round_num}/{rounds} ═══[/bold yellow]")
            
            results = self.validate_miners()
            self.print_results(results)
            
            if round_num < rounds:
                console.print(f"\n[dim]Next round in {interval} seconds...[/dim]")
                time.sleep(interval)
        
        console.print("\n[bold green]✓ Validation complete![/bold green]")
        
        # Final summary
        console.print("\n[bold cyan]Final Scores:[/bold cyan]")
        for miner_url, score in self.scores.items():
            status = "✓ Trusted" if score > 0.5 else "✗ Untrusted"
            color = "green" if score > 0.5 else "red"
            console.print(f"  {miner_url}: [{color}]{score:.2f} - {status}[/{color}]")


# ============================================================
# LIVE BITTENSOR MODE
# ============================================================

class BlindValidator:
    """Live Bittensor validator with honey pot verification."""
    
    def __init__(self, config=None):
        import bittensor as bt
        
        self.config = config or self.get_config()
        bt.logging(config=self.config)
        
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        
        self.fhe_client = None
        self.evaluation_keys = None
        self._load_fhe_client()
        
        self.scores = np.zeros(self.metagraph.n)
        
        console.print("[green]✓ Blind Validator initialized[/green]")
    
    def _load_fhe_client(self):
        """Load FHE client."""
        from concrete.ml.deployment import FHEModelClient
        
        model_path = Path(__file__).parent.parent / "fhe_models" / "credit_scorer"
        
        if (model_path / "client.zip").exists():
            self.fhe_client = FHEModelClient(str(model_path))
            self.evaluation_keys = self.fhe_client.get_serialized_evaluation_keys()
            console.print("[green]✓ Loaded FHE client[/green]")
    
    @staticmethod
    def get_config():
        import bittensor as bt
        
        parser = argparse.ArgumentParser(description="Dark Subnet Blind Validator")
        
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.dendrite.add_args(parser)
        bt.logging.add_args(parser)
        
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID")
        parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
        parser.add_argument("--miner-url", type=str, default="http://localhost:8091", 
                           help="Miner URL for simulation")
        parser.add_argument("--rounds", type=int, default=5, help="Validation rounds")
        
        return bt.config(parser)
    
    def create_honey_pot(self) -> tuple[bytes, int]:
        """Create a honey pot trap."""
        trap_profiles = [
            {"features": [0.95, 0.1, 0.9, 0.8, 0.1, 0.95, 0.1, 0.9, 0.0, 1.0], "expected": 1},
            {"features": [0.3, 0.9, 0.1, 0.2, 0.9, 0.15, 0.85, 0.05, 1.0, 0.0], "expected": 0},
        ]
        
        trap = random.choice(trap_profiles)
        features = np.array([trap["features"]], dtype=np.float32)
        encrypted = self.fhe_client.quantize_encrypt_serialize(features)
        
        return encrypted, trap["expected"]
    
    def run(self):
        """Main validation loop for live Bittensor."""
        import bittensor as bt
        from protocol.synapse import FHESynapse
        
        console.print("\n[bold green]═══ Blind Validator Running (Live) ═══[/bold green]")
        
        while True:
            try:
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                
                for uid in range(self.metagraph.n):
                    axon = self.metagraph.axons[uid]
                    if not axon.is_serving:
                        continue
                    
                    encrypted_trap, expected = self.create_honey_pot()
                    
                    synapse = FHESynapse(
                        encrypted_data=encrypted_trap,
                        evaluation_keys=self.evaluation_keys,
                        request_id=str(uuid.uuid4()),
                    )
                    
                    responses = self.dendrite.query(
                        axons=[axon],
                        synapse=synapse,
                        timeout=30.0,
                    )
                    
                    if responses and responses[0].encrypted_prediction:
                        result = self.fhe_client.deserialize_decrypt_dequantize(
                            responses[0].encrypted_prediction
                        )
                        actual = int(result[0])
                        
                        if actual == expected:
                            self.scores[uid] = 0.9 * self.scores[uid] + 0.1
                            bt.logging.info(f"✓ Miner {uid} PASSED")
                        else:
                            self.scores[uid] *= 0.5
                            bt.logging.warning(f"✗ Miner {uid} FAILED")
                
                time.sleep(12)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
                break
            except Exception as e:
                bt.logging.error(f"Error: {e}")
                time.sleep(12)


def main():
    """Entry point for the blind validator."""
    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Dark Subnet - Blind Validator          ║[/bold magenta]")
    console.print("[bold magenta]║   Honey Pot Verification System          ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
    
    if "--simulate" in sys.argv:
        # Parse options
        miner_url = "http://localhost:8091"
        rounds = 5
        
        for i, arg in enumerate(sys.argv):
            if arg == "--miner-url" and i + 1 < len(sys.argv):
                miner_url = sys.argv[i + 1]
            if arg == "--rounds" and i + 1 < len(sys.argv):
                rounds = int(sys.argv[i + 1])
        
        validator = SimulatedValidator(miner_urls=[miner_url])
        validator.run(rounds=rounds)
    else:
        try:
            import bittensor as bt
            validator = BlindValidator()
            validator.run()
        except ImportError:
            console.print("[red]Bittensor not installed.[/red]")
            console.print("[yellow]For simulation mode, use: python validator.py --simulate[/yellow]")
            console.print("[yellow]For live mode, install: pip install bittensor[/yellow]")


if __name__ == "__main__":
    main()
