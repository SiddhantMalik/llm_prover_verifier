# Proof Checker

This project is a simple proof verification tool written in Python. It is designed to help users verify mathematical proofs or logical statements using custom utility functions and a main verifier module.

## Features
- Verify proofs of logical statements
- Generate counterexample in case of unable to proof
- LLM integration to generate proofs automatically

## File Structure
- `verifier.py`: Main module for proof verification
- `openai_key.txt`: (Optional) API key for OpenAI integration or could set the environmental variable "OPENAI_API_KEY"
- `Readme.md` : Project documentation

## Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/SiddhantMalik/llm_prover_verifier
   cd proof_checker
   ```
2. **Install dependencies**
   This project uses only standard Python libraries. If you add external dependencies, list them in a `requirements.txt` file.
3. **Run the verifier**
   ```bash
   python verifier.py
   ```
4. **Run tests**
   Modify the premises and goal in the main() section of the `verifier.py`

## Usage
- Place your proof or logical statement in the appropriate format.
- Run `verifier.py` to check validity.
- Review results in the terminal.