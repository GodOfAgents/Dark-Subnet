"""
Blind Miner - FHE Inference Server

This miner performs inference on ENCRYPTED data without ever seeing the plaintext.
The miner receives FHE-encrypted inputs and returns FHE-encrypted predictions.

Key Innovation: The miner NEVER sees:
- The patient's age, medical history, or diagnosis
- The customer's income, credit score, or financial data
- Any personally identifiable information

All the miner sees is mathematical noise (ciphertext).

Usage:
    # Simulation mode (no Bittensor required)
    python miner.py --simulate
    
    # Live Bittensor mode
    python miner.py --netuid 1 --wallet.name miner --wallet.hotkey default
"""

import argparse
import time
import os
import sys
import socket
import json
import threading
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


# ============================================================
# SIMULATION MODE - No Bittensor Required
# ============================================================

class SimulatedMiner:
    """
    A simulated miner that runs without Bittensor network.
    
    This is for hackathon demos - shows the FHE logic working
    without needing wallet setup or network registration.
    """
    
    def __init__(self, port: int = 8091):
        self.port = port
        self.fhe_server = None
        self._load_fhe_model()
        
    def _load_fhe_model(self):
        """Load the compiled FHE model."""
        try:
            from concrete.ml.deployment import FHEModelServer
            
            model_path = Path(__file__).parent.parent / "fhe_models" / "credit_scorer"
            
            if (model_path / "server.zip").exists():
                self.fhe_server = FHEModelServer(str(model_path))
                console.print("[green]✓ Loaded FHE model: credit_scorer[/green]")
            else:
                console.print("[yellow]⚠ Model not compiled. Run: python fhe_models/train_model.py[/yellow]")
        except ImportError:
            console.print("[red]✗ Concrete ML not installed. Run: pip install concrete-ml[/red]")
    
    def process_request(self, encrypted_data: bytes, evaluation_keys: bytes) -> dict:
        """
        Process an encrypted inference request.
        
        BLIND INFERENCE: We compute on ciphertext without seeing plaintext.
        """
        start_time = time.time()
        
        if self.fhe_server is None:
            return {
                "success": False,
                "error": "FHE model not loaded",
                "encrypted_prediction": None,
                "computation_time_ms": 0,
            }
        
        try:
            # === THE BLIND INFERENCE ===
            # This is the core innovation - we process encrypted data
            # without EVER seeing what's inside
            
            console.print("\n[bold magenta]⚡ BLIND INFERENCE[/bold magenta]")
            console.print(f"[dim]Received {len(encrypted_data):,} bytes of ciphertext[/dim]")
            console.print("[yellow]Processing encrypted data (miner sees ONLY noise)...[/yellow]")
            
            encrypted_result = self.fhe_server.run(
                serialized_encrypted_quantized_data=encrypted_data,
                serialized_evaluation_keys=evaluation_keys,
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            console.print(f"[green]✓ Inference complete: {computation_time:.2f}ms[/green]")
            console.print(f"[dim]Returning {len(encrypted_result):,} bytes of encrypted result[/dim]")
            
            return {
                "success": True,
                "error": None,
                "encrypted_prediction": encrypted_result,
                "computation_time_ms": computation_time,
            }
            
        except Exception as e:
            console.print(f"[red]✗ Inference failed: {e}[/red]")
            return {
                "success": False,
                "error": str(e),
                "encrypted_prediction": None,
                "computation_time_ms": (time.time() - start_time) * 1000,
            }
    
    def run_server(self):
        """Run HTTP server for receiving requests."""
        miner = self
        
        class MinerHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default logging
            
            def do_POST(self):
                if self.path == "/inference":
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    try:
                        # Parse request
                        request = json.loads(post_data.decode('utf-8'))
                        encrypted_data = bytes.fromhex(request['encrypted_data'])
                        evaluation_keys = bytes.fromhex(request['evaluation_keys'])
                        
                        # Process
                        result = miner.process_request(encrypted_data, evaluation_keys)
                        
                        # Prepare response
                        response = {
                            "success": result["success"],
                            "error": result["error"],
                            "computation_time_ms": result["computation_time_ms"],
                        }
                        
                        if result["encrypted_prediction"]:
                            response["encrypted_prediction"] = result["encrypted_prediction"].hex()
                        
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                        
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode())
                
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "healthy",
                        "mode": "simulation",
                        "fhe_loaded": miner.fhe_server is not None,
                    }).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        server = HTTPServer(('localhost', self.port), MinerHandler)
        console.print(f"\n[bold green]═══ Simulated Miner Running ═══[/bold green]")
        console.print(f"  Mode: [cyan]SIMULATION[/cyan] (no Bittensor)")
        console.print(f"  Port: [cyan]{self.port}[/cyan]")
        console.print(f"  URL: [cyan]http://localhost:{self.port}[/cyan]")
        console.print(f"  FHE Model: [cyan]{'Loaded' if self.fhe_server else 'Not Loaded'}[/cyan]")
        console.print("\n[yellow]Waiting for encrypted requests...[/yellow]\n")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            server.shutdown()


# ============================================================
# LIVE BITTENSOR MODE
# ============================================================

class BlindMiner:
    """
    A miner that performs FHE inference without seeing plaintext data.
    
    This is the "blind" component of the Dark Subnet - the miner computes
    on encrypted data and has zero visibility into what it's processing.
    """
    
    def __init__(self, config=None):
        """Initialize the blind miner."""
        import bittensor as bt
        from protocol.synapse import FHESynapse, BatchFHESynapse
        
        self.config = config or self.get_config()
        
        # Setup logging
        bt.logging(config=self.config)
        
        # Setup wallet, subtensor, metagraph
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        
        # Setup axon (server)
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        
        # Load FHE servers for each model
        self.fhe_servers = {}
        self._load_fhe_models()
        
        # Attach handlers
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        console.print("[green]✓ Blind Miner initialized[/green]")
    
    def _load_fhe_models(self):
        """Load compiled FHE model servers."""
        from concrete.ml.deployment import FHEModelServer
        
        models_dir = Path(__file__).parent.parent / "fhe_models"
        
        # Load credit scorer model
        credit_scorer_path = models_dir / "credit_scorer"
        if credit_scorer_path.exists():
            server_zip = credit_scorer_path / "server.zip"
            if server_zip.exists():
                self.fhe_servers["credit_scorer"] = FHEModelServer(str(credit_scorer_path))
                console.print(f"[green]✓ Loaded FHE model: credit_scorer[/green]")
            else:
                console.print(f"[yellow]⚠ Model not compiled: {credit_scorer_path}[/yellow]")
                console.print("  Run: python fhe_models/train_model.py")
        else:
            console.print(f"[yellow]⚠ Model directory not found: {credit_scorer_path}[/yellow]")
    
    @staticmethod
    def get_config():
        """Create and return configuration."""
        import bittensor as bt
        
        parser = argparse.ArgumentParser(description="Dark Subnet Blind Miner")
        
        # Add bittensor args
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.axon.add_args(parser)
        bt.logging.add_args(parser)
        
        # Custom args
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID")
        parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
        parser.add_argument("--port", type=int, default=8091, help="Port for simulation server")
        
        return bt.config(parser)
    
    def forward(self, synapse):
        """Process a single FHE inference request."""
        from protocol.synapse import FHESynapse
        import bittensor as bt
        
        start_time = time.time()
        
        try:
            model_name = synapse.model_name or "credit_scorer"
            
            if model_name not in self.fhe_servers:
                synapse.error_message = f"Model not found: {model_name}"
                bt.logging.warning(f"Model not found: {model_name}")
                return synapse
            
            fhe_server = self.fhe_servers[model_name]
            
            # === BLIND INFERENCE ===
            encrypted_result = fhe_server.run(
                serialized_encrypted_quantized_data=synapse.encrypted_data,
                serialized_evaluation_keys=synapse.evaluation_keys,
            )
            
            synapse.encrypted_prediction = encrypted_result
            synapse.computation_time_ms = (time.time() - start_time) * 1000
            
            bt.logging.info(
                f"✓ Blind inference complete in {synapse.computation_time_ms:.2f}ms"
            )
            
        except Exception as e:
            synapse.error_message = str(e)
            bt.logging.error(f"FHE inference failed: {e}")
        
        return synapse
    
    def blacklist(self, synapse) -> tuple:
        """Determine if a request should be blacklisted."""
        caller_hotkey = synapse.dendrite.hotkey
        
        if caller_hotkey not in self.metagraph.hotkeys:
            return True, "Caller not registered"
        
        return False, ""
    
    def priority(self, synapse) -> float:
        """Calculate request priority based on caller's stake."""
        caller_hotkey = synapse.dendrite.hotkey
        
        if caller_hotkey in self.metagraph.hotkeys:
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            return float(self.metagraph.S[caller_uid])
        
        return 0.0
    
    def run(self):
        """Main loop - serve the axon and sync metagraph."""
        import bittensor as bt
        
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        
        console.print("\n[bold green]═══ Blind Miner Running (Live) ═══[/bold green]")
        console.print(f"  Hotkey: {self.wallet.hotkey.ss58_address}")
        console.print(f"  Axon: {self.axon.info()}")
        
        step = 0
        while True:
            try:
                if step % 100 == 0:
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                    bt.logging.info(f"Synced metagraph, block: {self.metagraph.block}")
                
                step += 1
                time.sleep(12)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
                self.axon.stop()
                break
            except Exception as e:
                bt.logging.error(f"Error in main loop: {e}")
                time.sleep(12)


def main():
    """Entry point for the blind miner."""
    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Dark Subnet - Blind Miner              ║[/bold magenta]")
    console.print("[bold magenta]║   FHE Inference on Encrypted Data        ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
    
    # Check for simulation mode
    if "--simulate" in sys.argv:
        # Parse port if provided
        port = 8091
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        
        miner = SimulatedMiner(port=port)
        miner.run_server()
    else:
        # Live Bittensor mode
        try:
            import bittensor as bt
            miner = BlindMiner()
            miner.run()
        except ImportError:
            console.print("[red]Bittensor not installed.[/red]")
            console.print("[yellow]For simulation mode, use: python miner.py --simulate[/yellow]")
            console.print("[yellow]For live mode, install: pip install bittensor[/yellow]")


if __name__ == "__main__":
    main()
