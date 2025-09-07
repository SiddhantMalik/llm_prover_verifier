# Proof Checker

This project is a simple proof verification tool written in Python. It is designed to help users verify mathematical proofs or logical statements using custom utility functions and a main verifier module.

## Features
- Verify proofs or logical statements
- Utility functions for parsing and checking
- Easy to use and extend

## File Structure
- `verifier.py`: Main module for proof verification
- `utils.py`: Utility functions used by the verifier
- `tests.py`: Test cases for the verifier and utilities
- `openai_key.txt`: (Optional) API key for OpenAI integration (if used)
- `Readme.md` / `readme.txt`: Project documentation

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
   ```bash
   python tests.py
   ```

## Usage
- Place your proof or logical statement in the appropriate format.
- Run `verifier.py` to check validity.
- Review results in the terminal.

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is open source and available under the MIT License.
