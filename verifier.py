
import os
import re
from openai import OpenAI

# LLM class for proof generation and repair
class LLM:
	def __init__(self, api_key=None):
		"""
		Loads OpenAI API key from environment variable OPENAI_API_KEY or local file 'openai_key.txt'.
		Remove your API key before submission!
		"""
		if api_key:
			self.api_key = api_key
		elif 'OPENAI_API_KEY' in os.environ:
			self.api_key = os.environ['OPENAI_API_KEY']
		else:
			try:
				with open('openai_key.txt') as f:
					self.api_key = f.read().strip()
			except Exception:
				raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or create openai_key.txt.")
		self.client = OpenAI(api_key=self.api_key)

	def generate_proof(self, premises, goal):
		"""
		Accepts premises (list of strings) and goal (string).
		Returns proof as list of lines: [(number, formula, justification)]
		"""
		prompt = self._build_prompt(premises, goal)
		response = self.client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[{"role": "user", "content": prompt}]
		)
		proof_text = response.choices[0].message.content
		return self._parse_proof(proof_text)

	def _build_prompt(self, premises, goal):
		return (
			"You are a proof assistant for propositional logic using only the Lukasiewicz-Church (P2) axiom system.\n"
			"Axiom Schemas:\n"
			"AX1: A → (B → A)\n"
			"AX2: (A → (B → C)) → ((A → B) → (A → C))\n"
			"AX3: (¬B → ¬A) → (A → B)\n"
			"Rule of Inference:\n"
			"Modus Ponens (MP): From φ and φ → ψ, infer ψ.\n"
			"Format: <number>: <formula> [<justification>]\n"
			"Justifications: Premise, AX1, AX2, AX3, MP i, j\n"
			"Example 1:\n"
			"Premises: P → Q, P\n"
			"Goal: Q\n"
			"Proof:\n"
			"1: (P → Q) [Premise]\n"
			"2: P [Premise]\n"
			"3: Q [MP 2, 1]\n"
			"Example 2:\n"
			"Premises: P → (Q → P)\n"
			"Goal: Q → P\n"
			"Proof:\n"
			"1: (P → (Q → P)) [Premise]\n"
			"2: (Q → P) [MP 1, 1]\n"
			"Now, given the premises and goal below, generate a numbered proof using only the above rules.\n"
			f"Premises: {', '.join(premises)}\nGoal: {goal}\nProof:"
		)

	def _parse_proof(self, proof_text):
		# Parse proof lines from LLM output
		lines = []
		for line in proof_text.strip().split('\n'):
			m = re.match(r'(\d+):\s*(.+)\s*\[(.+)\]', line)
			if m:
				number = int(m.group(1))
				formula = parse_formula(m.group(2))
				justification = m.group(3)
				lines.append(ProofLine(number, formula, justification))
		return lines

	def repair_proof(self, premises, goal, previous_proof, error_msg):
		# Prompt LLM to revise proof
		prev_text = '\n'.join([
			f"{line.number}: {line.formula} [{line.justification}]" for line in previous_proof
		])
		prompt = (
			"Your previous proof was invalid. Error: " + error_msg + "\n"
			"Premises: " + ', '.join(premises) + "\nGoal: " + goal + "\n"
			"Previous proof:\n" + prev_text + "\n"
			"Please revise the proof to fix the error, using only Modus Ponens and axioms AX1, AX2, AX3."
		)
		response = self.client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[{"role": "user", "content": prompt}]
		)
		proof_text = response.choices[0].message.content
		return self._parse_proof(proof_text)

# Orchestrator for LLM-assisted proof generation and verification
class LLMOrchestrator:
	def __init__(self, llm):
		self.llm = llm

	def run(self, premises, goal, max_attempts=5):
		for attempt in range(max_attempts):
			proof_lines = self.llm.generate_proof(premises, goal)
			print(f"Attempt {attempt+1}: Generated proof:")
			for line in proof_lines:
				print(line)
			verifier = ProofVerifier(proof_lines)
			try:
				valid = verifier.verify()
			except Exception as e:
				valid = False
				error_msg = str(e)
			if valid:
				print("VALID PROOF FOUND:")
				return True, proof_lines
			else:
				print(f"Attempt {attempt+1}: Proof invalid. Repairing...")
				error_msg = "Proof invalid. See above."
				proof_lines = self.llm.repair_proof(premises, goal, proof_lines, error_msg)
		print("FAILED: No valid proof found.")
		print("Last attempted proof:")
		for line in proof_lines:
			print(line)
		return False, proof_lines


# Formula representation: variable, negation, implication
class Formula:
	def __init__(self, value, left=None, right=None):
		self.value = value  # 'var', 'not', 'imp'
		self.left = left    # For 'not', left is the subformula; for 'imp', left/right are subformulas
		self.right = right

	def __eq__(self, other):
		if not isinstance(other, Formula):
			return False
		return (self.value == other.value and
				self.left == other.left and
				self.right == other.right)

	def __repr__(self):
		if self.value == 'var':
			return self.left
		elif self.value == 'not':
			return f"¬{self.left}"
		elif self.value == 'imp':
			return f"({self.left} → {self.right})"
		return "?"

def parse_formula(s):
	# Remove spaces
	s = s.replace(' ', '')
	# Negation
	if s.startswith('¬'):
		return Formula('not', parse_formula(s[1:]))
	# Implication: look for top-level '->' or '→'
	depth = 0
	for i, c in enumerate(s):
		if c == '(':
			depth += 1
		elif c == ')':
			depth -= 1
		elif (c == '-' and i+1 < len(s) and s[i+1] == '>') or c == '→':
			if depth == 0:
				if c == '-':
					left = s[:i]
					right = s[i+2:]
				else:
					left = s[:i]
					right = s[i+1:]
				return Formula('imp', parse_formula(left), parse_formula(right))
	# Parentheses
	if s.startswith('(') and s.endswith(')'):
		return parse_formula(s[1:-1])
	# Variable
	return Formula('var', s)

class ProofLine:
	def __init__(self, number, formula, justification):
		self.number = number
		self.formula = formula
		self.justification = justification

	def __repr__(self):
		return f"{self.number}: {self.formula} [{self.justification}]"

# Axiom schema matchers

# Utility: match axiom schema with variable substitution
def match_schema(schema, formula, mapping=None):
	if mapping is None:
		mapping = {}
	if schema.value == 'var':
		var = schema.left
		if var in mapping:
			return formula == mapping[var]
		else:
			mapping[var] = formula
			return True
	if schema.value != formula.value:
		return False
	if schema.value == 'not':
		return match_schema(schema.left, formula.left, mapping)
	if schema.value == 'imp':
		return match_schema(schema.left, formula.left, mapping) and match_schema(schema.right, formula.right, mapping)
	return False

def ax1_schema():
	# AX1: A → (B → A)
	A = Formula('var', 'A')
	B = Formula('var', 'B')
	return Formula('imp', A, Formula('imp', B, A))

def ax2_schema():
	# AX2: (A → (B → C)) → ((A → B) → (A → C))
	A = Formula('var', 'A')
	B = Formula('var', 'B')
	C = Formula('var', 'C')
	left = Formula('imp', A, Formula('imp', B, C))
	right = Formula('imp', Formula('imp', A, B), Formula('imp', A, C))
	return Formula('imp', left, right)

def ax3_schema():
	# AX3: (¬B → ¬A) → (A → B)
	A = Formula('var', 'A')
	B = Formula('var', 'B')
	left = Formula('imp', Formula('not', B), Formula('not', A))
	right = Formula('imp', A, B)
	return Formula('imp', left, right)

def match_ax1(formula):
	return match_schema(ax1_schema(), formula)

def match_ax2(formula):
	return match_schema(ax2_schema(), formula)

def match_ax3(formula):
	return match_schema(ax3_schema(), formula)

class ProofVerifier:
	def __init__(self, proof_lines):
		self.proof_lines = proof_lines  # List of ProofLine
		self.line_map = {line.number: line for line in proof_lines}

	def verify(self):
		for line in self.proof_lines:
			just = line.justification
			if just.lower() == 'premise':
				continue
			elif just.upper() == 'AX1':
				if not match_ax1(line.formula):
					print(f"Line {line.number} is not an instance of AX1.")
					return False
			elif just.upper() == 'AX2':
				if not match_ax2(line.formula):
					print(f"Line {line.number} is not an instance of AX2.")
					return False
			elif just.upper() == 'AX3':
				if not match_ax3(line.formula):
					print(f"Line {line.number} is not an instance of AX3.")
					return False
			elif just.startswith('MP'):
				# Modus Ponens: MP i, j
				m = re.match(r'MP\s*(\d+),\s*(\d+)', just)
				if not m:
					print(f"Line {line.number}: Invalid MP format.")
					return False
				i, j = int(m.group(1)), int(m.group(2))
				if i not in self.line_map or j not in self.line_map:
					print(f"Line {line.number}: MP references missing lines.")
					return False
				phi = self.line_map[i].formula
				imp = self.line_map[j].formula
				if imp.value != 'imp' or imp.left != phi or imp.right != line.formula:
					print(f"Line {line.number}: MP does not match formulas.")
					return False
			else:
				print(f"Line {line.number}: Unknown justification '{just}'.")
				return False
		print("Proof is valid.")
		return True

# Example usage
if __name__ == "__main__":
	# Example: LLM-assisted proof generation and verification
	# Premises and goal
	premises = ['P → Q', 'Q → R']
	goal = 'P → R'

	# Initialize LLM and orchestrator
	llm = LLM()
	orchestrator = LLMOrchestrator(llm)

	# Run proof generation and verification
	valid, proof_lines = orchestrator.run(premises, goal)

	print("Final outcome:", "VALID" if valid else "FAILED")
	print("Proof:")
	for line in proof_lines:
		print(line)

