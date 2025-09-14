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
		# Maintain conversation history for iterative proof improvement
		self.conversation_history = []
	
	def reset_conversation(self):
		"""Reset conversation history for a new proof problem."""
		self.conversation_history = []

	def generate_proof(self, premises, goal):
		"""
		Accepts premises (list of strings) and goal (string).
		Returns proof as list of lines: [(number, formula, justification)]
		"""
		prompt = self._build_prompt(premises, goal)
		
		# Initialize conversation history if this is the first attempt
		if not self.conversation_history:
			self.conversation_history = [{"role": "user", "content": prompt}]
		
		response = self.client.chat.completions.create(
			model="gpt-4o-mini",
			messages=self.conversation_history
		)
		proof_text = response.choices[0].message.content
		
		# Add the response to conversation history
		self.conversation_history.append({"role": "assistant", "content": proof_text})
		
		return self._parse_proof(proof_text)

	def _build_prompt(self, premises, goal):
		return (
			"You are a professional proof solver for propositional logic using only the Lukasiewicz-Church (P2) axiom system.\n"
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

	def formula_equiv(self, f1, f2):
		"""
		Checks if two formulas are equivalent by structure and value.
		This can be extended for more rigorous logical equivalence if needed.
		"""
		if f1 == f2:
			return True
		# Remove unnecessary parentheses and spaces for comparison
		def normalize(formula):
			if isinstance(formula, Formula):
				if formula.value == 'var':
					return formula.left.replace(' ', '')
				elif formula.value == 'not':
					return '¬' + normalize(formula.left)
				elif formula.value == 'imp':
					return f"({normalize(formula.left)}→{normalize(formula.right)})"
			return str(formula).replace(' ', '')
		return normalize(f1) == normalize(f2)

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

	def repair_proof(self, premises, goal, previous_proof, error_msg, detailed_error_analysis=None):
		# Build a comprehensive error feedback message
		prev_text = '\n'.join([
			f"{line.number}: {line.formula} [{line.justification}]" for line in previous_proof
		])
		
		# Create detailed feedback message
		feedback_msg = f"Your previous proof was invalid. Here's what went wrong:\n\nError: {error_msg}\n"
		
		if detailed_error_analysis:
			feedback_msg += f"\nDetailed Analysis:\n{detailed_error_analysis}\n"
		
		feedback_msg += f"\nOriginal Problem:\nPremises: {', '.join(premises)}\nGoal: {goal}\n"
		feedback_msg += f"\nYour Previous Attempt:\n{prev_text}\n"
		feedback_msg += "\nPlease analyze the error carefully and provide a corrected proof. "
		feedback_msg += "Remember to:\n"
		feedback_msg += "1. Use only the axioms AX1, AX2, AX3 and Modus Ponens (MP)\n"
		feedback_msg += "2. Ensure each line is properly justified\n"
		feedback_msg += "3. Check that MP applications match the correct formula patterns\n"
		feedback_msg += "4. Make sure the final line matches the goal exactly\n\n"
		feedback_msg += "Provide the corrected proof in the same format: <number>: <formula> [<justification>]"
		
		# Add the feedback to conversation history
		self.conversation_history.append({"role": "user", "content": feedback_msg})
		
		response = self.client.chat.completions.create(
			model="gpt-4o-mini",  
			messages=self.conversation_history
		)
		proof_text = response.choices[0].message.content
		
		# Add the response to conversation history
		self.conversation_history.append({"role": "assistant", "content": proof_text})
		
		return self._parse_proof(proof_text)

	def generate_counterexample(self, premises, goal):
		"""
		Prompts the LLM to provide a counterexample for the given premises and goal.
		Returns a string describing the counterexample, or None if not found.
		"""
		prompt = (
			"The statement below may not be valid. If it is not, provide a counterexample: "
			f"Premises: {', '.join(premises)}\nGoal: {goal}\n"
			"If the statement is valid, reply 'No counterexample exists.' Otherwise, describe a counterexample (an assignment of truth values to variables that makes all premises true and the goal false)."
		)
		response = self.client.chat.completions.create(
			model="gpt-5-mini",
			messages=[{"role": "user", "content": prompt}]
		)
		result = response.choices[0].message.content.strip()
		if result.lower().startswith("no counterexample"):
			return None
		return result


# Orchestrator for LLM-assisted proof generation and verification
class LLMOrchestrator:
	def __init__(self, llm):
		self.llm = llm

	def run(self, premises, goal, max_attempts=3):
		# Reset conversation history for this new problem
		self.llm.reset_conversation()
		
		for attempt in range(max_attempts):
			if attempt == 0:
				proof_lines = self.llm.generate_proof(premises, goal)
			else:
				# For subsequent attempts, the proof_lines will come from repair_proof
				pass
			
			print(f"Attempt {attempt+1}: Generated proof:")
			for line in proof_lines:
				print(line)
			
			verifier = ProofVerifier(proof_lines)
			try:
				valid, error_msg, detailed_analysis = verifier.verify(goal)
			except Exception as e:
				valid = False
				error_msg = str(e)
				detailed_analysis = f"Exception occurred during verification: {str(e)}"
				
			if valid and error_msg is None:
				print("VALID PROOF FOUND:")
				return True, proof_lines
			else:
				print(f"Attempt {attempt+1}: Proof invalid. Repairing...")
				print(f"Error: {error_msg}")
				if detailed_analysis:
					print(f"Detailed Analysis: {detailed_analysis}")
				
				# Don't repair on the last attempt
				if attempt < max_attempts - 1:
					proof_lines = self.llm.repair_proof(premises, goal, proof_lines, error_msg, detailed_analysis)
		
		print("FAILED: No valid proof found.")
		print("Last attempted proof:")
		for line in proof_lines:
			print(line)

		# Extra credit: Try to generate a counterexample using the LLM
		print("Attempting to generate a counterexample...")
		counterexample = self.llm.generate_counterexample(premises, goal)
		if counterexample:
			print("COUNTEREXAMPLE FOUND:")
			print(counterexample)
		else:
			print("No counterexample could be generated.")
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

	def verify(self, goal=None):
		if not self.proof_lines or len(self.proof_lines) == 0:
			print("Proof is empty.")
			return False, "Proof is empty.", "The proof contains no lines. You need to provide at least one line to prove the goal."
		
		detailed_analysis = []
		
		for line in self.proof_lines:
			just = line.justification
			if just.lower() == 'premise':
				continue
			elif just.upper() == 'AX1':
				if not match_ax1(line.formula):
					error_msg = f"Line {line.number}: Formula {line.formula} is not an instance of AX1."
					analysis = f"AX1 has the form A → (B → A). Your formula {line.formula} does not match this pattern. Check if you have the correct structure with proper variable substitutions."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
			elif just.upper() == 'AX2':
				if not match_ax2(line.formula):
					error_msg = f"Line {line.number}: Formula {line.formula} is not an instance of AX2."
					analysis = f"AX2 has the form (A → (B → C)) → ((A → B) → (A → C)). Your formula {line.formula} does not match this pattern. Ensure you have the correct nested implication structure."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
			elif just.upper() == 'AX3':
				if not match_ax3(line.formula):
					error_msg = f"Line {line.number}: Formula {line.formula} is not an instance of AX3."
					analysis = f"AX3 has the form (¬B → ¬A) → (A → B). Your formula {line.formula} does not match this pattern. Check that you have the correct contrapositive structure with proper negations."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
			elif just.startswith('MP'):
				m = re.match(r'MP\s*(\d+),\s*(\d+)', just)
				if not m:
					error_msg = f"Line {line.number}: Invalid MP format in justification '{just}'."
					analysis = f"Modus Ponens should be written as 'MP i, j' where i and j are line numbers. Your format '{just}' is incorrect."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
				i, j = int(m.group(1)), int(m.group(2))
				if i not in self.line_map or j not in self.line_map:
					error_msg = f"Line {line.number}: MP references missing lines {i} or {j}."
					analysis = f"You referenced lines {i} and {j} in your MP justification, but one or both of these lines don't exist in your proof. Check your line numbering."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
				phi = self.line_map[i].formula
				imp = self.line_map[j].formula
				if imp.value != 'imp' or imp.left != phi or imp.right != line.formula:
					error_msg = (f"Line {line.number}: MP does not match formulas. "
								f"Expected {phi} and {phi} → {line.formula}, but got {phi} and {imp}.")
					analysis = f"For MP to work, you need φ and φ → ψ to infer ψ. You have:\nLine {i}: {phi}\nLine {j}: {imp}\nBut line {j} should be {phi} → {line.formula} for MP to produce {line.formula}."
					detailed_analysis.append(analysis)
					print(error_msg)
					return False, error_msg, '\n'.join(detailed_analysis)
			else:
				error_msg = f"Line {line.number}: Unknown justification '{just}'."
				analysis = f"'{just}' is not a valid justification. Valid justifications are: Premise, AX1, AX2, AX3, or MP i, j (where i and j are line numbers)."
				detailed_analysis.append(analysis)
				print(error_msg)
				return False, error_msg, '\n'.join(detailed_analysis)
		
		# Check that the last line matches the goal
		if goal is not None:
			last_formula = self.proof_lines[-1].formula
			goal_formula = parse_formula(goal)
			if not Formula.__eq__(last_formula, goal_formula):
				error_msg = f"Proof does not reach the goal: {goal}"
				analysis = f"Your proof ends with {last_formula}, but the goal is {goal_formula}. The final line of your proof must exactly match the goal."
				detailed_analysis.append(analysis)
				print(error_msg)
				return False, error_msg, '\n'.join(detailed_analysis)
		
		print("Proof is valid.")
		return True, None, None
	
# Example usage
if __name__ == "__main__":
	# Example: LLM-assisted proof generation and verification
	# Premises and goal
	# premises = []
	# goal = []
 
	premises = [
		'P → (Q → R)',
		'S → P',
		'T → Q',
		'R → U',
		'¬U → ¬S'
		]
	goal = 'S → (T → U)'

	# Initialize LLM and orchestrator
	llm = LLM()
	orchestrator = LLMOrchestrator(llm)


	# Run proof generation and verification
	valid, proof_lines = orchestrator.run(premises, goal)

	if valid:
		print("Final outcome: VALID")
		print("Proof:")
		for line in proof_lines:
			print(line)
	else:
		print("Final outcome: FAILED")
		print("No valid proof found.")
		# Try to generate a counterexample and print it
		counterexample = llm.generate_counterexample(premises, goal)
		if counterexample:
			print("Counterexample:")
			print(counterexample)
		else:
			print("No Counterexample can be generated")