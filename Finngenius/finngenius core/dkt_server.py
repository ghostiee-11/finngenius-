import random
import time
import math
import torch
import torch.nn as nn
import os
from groq import Groq
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Data Structures (Updated Question Class) ---
class Concept:
    def __init__(self, id, name, prerequisites=None):
        self.id = id
        self.name = name
        self.prerequisites = prerequisites if prerequisites else []

class Question:
    # MODIFIED: Added options, changed correct_answer to correct_option
    def __init__(self, id, concept_id, text, options, correct_option):
        self.id = id
        self.concept_id = concept_id
        self.text = text
        self.options = options # e.g., {'A': 'Option 1', 'B': 'Option 2'}
        self.correct_option = correct_option # e.g., 'A'

class Module:
    def __init__(self, id, name, concepts_covered, content_sections):
        self.id = id
        self.name = name
        self.concepts_covered = concepts_covered
        self.content_sections = content_sections
        self.practice_questions = [] # List of question IDs

class User:
    def __init__(self, id, name, age):
        self.id = id
        self.name = name
        self.age = age
        self.interaction_history = []
        self.knowledge_state = {}
        self.current_roadmap = []
        self.completed_modules = set()
        self.learning_goal = None
        self.current_quiz_index = 0
        self.initial_quiz_ids = []
        self.current_module_id = None
        self.current_module_practice_index = 0
        self.state = "INITIALIZING"

class Interaction:
    def __init__(self, user_id, item_id, item_type, concept_id, correct, timestamp, selected_option=None): # Added selected_option
        self.user_id = user_id
        self.item_id = item_id
        self.item_type = item_type
        self.concept_id = concept_id
        self.correct = correct
        self.timestamp = timestamp
        self.selected_option = selected_option # Store which option was chosen

# --- LSTM-based DKT Model (Same as Before) ---
class LSTMDKTModel(nn.Module):
    def __init__(self, num_concepts, embedding_dim=50, hidden_dim=100, num_layers=1):
        super(LSTMDKTModel, self).__init__()
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.concept_embedding = nn.Embedding(num_concepts + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, concept_ids, responses):
        device = concept_ids.device
        embedded_concepts = self.concept_embedding(concept_ids)
        responses = responses.to(device)
        # Use a device-agnostic way to create the embedding layer if needed,
        # or ensure it's moved to the correct device when the model is initialized.
        # Simplest is often to move the layer in init if device is known, or here:
        response_embedding_layer = nn.Embedding(2, self.embedding_dim).to(device)
        embedded_responses = response_embedding_layer(responses)

        lstm_input = torch.cat((embedded_concepts, embedded_responses), dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        predictions = self.output_layer(lstm_out)
        return self.sigmoid(predictions)

# --- FinEdu System Logic (Modified for Multiple Choice) ---
class FinEduSystem:
    def __init__(self, concepts, questions, modules, dkt_model_class, groq_api_key=None):
        self.concepts = {c.id: c for c in concepts}
        self.questions = {q.id: q for q in questions}
        self.modules = {m.id: m for m in modules}
        self.concept_to_index = {concept.id: i for i, concept in enumerate(concepts)}
        self.index_to_concept = {i: concept.id for i, concept in enumerate(concepts)}
        self.num_concepts = len(concepts)
        self.dkt_model = dkt_model_class(self.num_concepts)
        self.dkt_model.eval()

        # Groq client init (same as before, keep for potential future use)
        self.groq_api_key = groq_api_key
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("Groq API client initialized (though not used for MC quiz check).")
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                self.groq_client = None
        else:
            self.groq_client = None
            print("Groq API key not provided.")

    def _find_module_for_concept(self, concept_id):
        for module_id, module in self.modules.items():
            if concept_id in module.concepts_covered:
                return module
        return None

    def start_session_for_user(self, user, initial_quiz_ids, learning_goal_concept_id):
        print(f"--- Starting FinEdu Session for {user.name} (ID: {user.id}) ---")
        user.learning_goal = learning_goal_concept_id
        user.knowledge_state = {concept_id: 0.5 for concept_id in self.concepts}
        user.initial_quiz_ids = initial_quiz_ids
        user.current_quiz_index = 0
        user.state = "QUIZ"
        print(f"User {user.id} starting quiz. Goal: {learning_goal_concept_id}")
        return self.get_next_item_for_user(user)

    # MODIFIED: Simplified answer check for multiple choice
    def _check_answer(self, question, selected_option_key):
        """Checks if the selected option key matches the correct option key."""
        is_correct = (selected_option_key == question.correct_option)
        print(f"MC Check: Q: '{question.text[:50]}...' Selected: '{selected_option_key}', Correct: '{question.correct_option}' -> Result: {is_correct}")
        return is_correct

    def process_user_answer(self, user, question_id, item_type, selected_option_key):
        """Processes a user's selected option, updates state, and determines the next item."""
        question = self.questions.get(question_id)
        if not question:
            print(f"Error: Question {question_id} not found for user {user.id}")
            return {"error": "Question not found"}, 404
        if not hasattr(question, 'correct_option'): # Check if it's an MC question
             print(f"Error: Question {question_id} is not a multiple-choice question.")
             return {"error": "Invalid question type for this answer format"}, 400

        is_correct = self._check_answer(question, selected_option_key)
        print(f"User {user.id} answered Q{question_id} ({item_type}) with option '{selected_option_key}': {'Correct' if is_correct else 'Incorrect'}")

        # Record interaction
        interaction = Interaction(
            user_id=user.id, item_id=question_id, item_type=item_type,
            concept_id=question.concept_id, correct=is_correct, timestamp=time.time(),
            selected_option=selected_option_key # Store the selected option
        )
        user.interaction_history.append(interaction)

        # Update DKT
        concept_index = self.concept_to_index.get(question.concept_id)
        if concept_index is not None:
            self._update_knowledge_state_lstm(user, [concept_index], [1 if is_correct else 0])
        else:
             print(f"Warning: Concept {question.concept_id} not found in index for DKT update.")

        # MODIFIED: Include correct_option in feedback for frontend button styling
        feedback = {
            "correct": is_correct,
            "selected_option": selected_option_key,
            "correct_option": question.correct_option # Send correct option back
            }

        # --- Determine next state (Logic remains largely the same) ---
        next_item = None
        if user.state == "QUIZ":
            user.current_quiz_index += 1
            if user.current_quiz_index < len(user.initial_quiz_ids):
                next_item = self.get_next_item_for_user(user)
            else:
                print(f"User {user.id} finished quiz. Generating roadmap.")
                self.generate_roadmap(user)
                user.state = "LEARNING"
                next_item = self.get_next_item_for_user(user)

        elif user.state == "LEARNING":
            module = self.modules.get(user.current_module_id)
            module_practice_done = (not module) or (user.current_module_practice_index >= len(module.practice_questions))

            if module and not module_practice_done:
                 next_item = self.get_next_item_for_user(user)
            else:
                if module:
                    print(f"User {user.id} finished practice for module {user.current_module_id}.")
                    module_mastered = all(self.predict_mastery(user, cid) for cid in module.concepts_covered)
                    if module_mastered:
                        print(f"Module {module.id} mastered by user {user.id}.")
                        user.completed_modules.add(user.current_module_id)
                    else:
                        print(f"Module {module.id} not mastered, adding back to roadmap.")
                        if user.current_module_id not in user.current_roadmap:
                             user.current_roadmap.append(user.current_module_id)

                if self.check_goal_completion(user):
                    user.state = "GOAL_REACHED"
                    print(f"*** User {user.id} achieved learning goal! ***")
                    next_item = self.get_next_item_for_user(user)
                else:
                    user.current_module_id = None
                    user.current_module_practice_index = 0
                    next_item = self.get_next_item_for_user(user)
        else:
             print(f"Warning: User {user.id} in unexpected state '{user.state}' during answer processing.")
             user.state = "LEARNING"
             next_item = self.get_next_item_for_user(user)

        return {"feedback": feedback, "next_item": next_item}

    # MODIFIED: Include options and correct_option in question response
    def get_next_item_for_user(self, user):
        """Determines and returns the next learning item based on user state."""
        print(f"Get next item for User {user.id}, State: {user.state}, Current Module: {user.current_module_id}, Roadmap: {user.current_roadmap}")

        if user.state == "QUIZ":
            if user.current_quiz_index < len(user.initial_quiz_ids):
                q_id = user.initial_quiz_ids[user.current_quiz_index]
                question = self.questions.get(q_id)
                if question and hasattr(question, 'options') and hasattr(question, 'correct_option'):
                    print(f"  -> Next: Quiz Question {q_id}")
                    return {
                        "type": "quiz_question",
                        "item_id": q_id,
                        "text": question.text,
                        "options": question.options, # Send options
                        "correct_option": question.correct_option, # Send correct key
                        "concept_id": question.concept_id
                    }
                else:
                    print(f"  -> Skipping invalid/non-MC quiz question ID: {q_id}")
                    user.current_quiz_index += 1
                    return self.get_next_item_for_user(user)
            else:
                print(f"Error: get_next_item called in QUIZ state but quiz index is out of bounds. Transitioning.")
                user.state = "LEARNING"
                # Fall through

        if user.state == "LEARNING":
            # Check practice questions first
            if user.current_module_id:
                module = self.modules.get(user.current_module_id)
                if module and user.current_module_practice_index < len(module.practice_questions):
                    q_id = module.practice_questions[user.current_module_practice_index]
                    question = self.questions.get(q_id)
                    if question and hasattr(question, 'options') and hasattr(question, 'correct_option'):
                        practice_index_to_send = user.current_module_practice_index
                        user.current_module_practice_index += 1
                        print(f"  -> Next: Practice Question {q_id} (Index {practice_index_to_send}) for Module {module.id}")
                        return {
                            "type": "practice_question",
                            "item_id": q_id,
                            "text": question.text,
                            "options": question.options, # Send options
                            "correct_option": question.correct_option, # Send correct key
                            "concept_id": question.concept_id,
                            "module_id": module.id,
                        }
                    else:
                        print(f"  -> Skipping invalid/non-MC practice question ID: {q_id} in Module {module.id}")
                        user.current_module_practice_index += 1
                        return self.get_next_item_for_user(user)
                # If no more practice questions or invalid module, fall through to get next module

            # If no current module or practice done, get next from roadmap
            user.current_module_id = None
            user.current_module_practice_index = 0
            if user.current_roadmap:
                initial_roadmap_len = len(user.current_roadmap)
                attempts = 0
                while attempts < initial_roadmap_len:
                    next_module_id = user.current_roadmap.pop(0)
                    module = self.modules.get(next_module_id)
                    if module:
                        user.current_module_id = module.id
                        user.current_module_practice_index = 0
                        print(f"  -> Next: Module Content {module.id} ('{module.name}')")
                        return {
                            "type": "module_content",
                            "item_id": module.id,
                            "name": module.name,
                            "concepts": [self.concepts.get(c, Concept(c, "Unknown")).name for c in module.concepts_covered],
                            "content_sections": module.content_sections
                        }
                    else:
                        print(f"  -> Skipping invalid module ID {next_module_id} from roadmap.")
                        attempts += 1
                print(f"  -> Exhausted roadmap attempts, remaining items were invalid.")
                user.current_roadmap = []

            # If roadmap is empty
            user.state = "FINISHED_ROADMAP"
            print(f"  -> Roadmap empty. Transitioning to FINISHED_ROADMAP.")
            # Fall through

        if user.state == "GOAL_REACHED":
             goal_concept_name = self.concepts[user.learning_goal].name if user.learning_goal in self.concepts else "Unknown Goal"
             print(f"  -> Next: Goal Reached Message")
             return { "type": "goal_reached", "message": f"Congratulations {user.name}! You've demonstrated mastery of '{goal_concept_name}'.", "knowledge_state": {self.concepts[cid].name: round(prob, 2) for cid, prob in user.knowledge_state.items()}}

        if user.state == "FINISHED_ROADMAP":
             goal_concept_name = self.concepts[user.learning_goal].name if user.learning_goal in self.concepts else "Unknown Goal"
             goal_mastered = self.check_goal_completion(user)
             message = f"You've completed the current learning path."
             if goal_mastered: message += f" You also achieved the goal: '{goal_concept_name}'!"
             else: message += f" The learning goal '{goal_concept_name}' may require further study."
             print(f"  -> Next: Finished Roadmap Message")
             return { "type": "finished_roadmap", "message": message, "knowledge_state": {self.concepts[cid].name: round(prob, 2) for cid, prob in user.knowledge_state.items()}}

        print(f"Error: Could not determine next item for user {user.id} in state {user.state}")
        return {"type": "error", "message": "Could not determine next action."}

    # --- DKT Update and other methods (generate_roadmap, predict_mastery, etc.) ---
    # --- Keep these methods largely the same as before ---

    def _update_knowledge_state_lstm(self, user, concept_indices, responses):
        """Updates the user's knowledge state using the LSTM model."""
        if not concept_indices or not responses: return

        self.dkt_model.eval()
        with torch.no_grad():
            concept_ids_tensor = torch.tensor(concept_indices, dtype=torch.long).unsqueeze(0)
            responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

            device = next(self.dkt_model.parameters()).device
            concept_ids_tensor = concept_ids_tensor.to(device)
            # responses_tensor are handled in forward method's embedding layer move

            predictions = self.dkt_model(concept_ids_tensor, responses_tensor)

            last_prediction = predictions[0, -1, :].cpu().numpy()
            for i in range(self.num_concepts):
                concept_id = self.index_to_concept.get(i)
                if concept_id:
                    user.knowledge_state[concept_id] = float(last_prediction[i])

        if concept_indices:
             last_concept_idx = concept_indices[-1]
             last_concept_id = self.index_to_concept.get(last_concept_idx)
             if last_concept_id:
                  print(f"DKT (LSTM): Updated User {user.id}. Concept '{self.concepts[last_concept_id].name}' new prob: {user.knowledge_state.get(last_concept_id, 0):.3f}")

    def generate_roadmap(self, user):
        """Generates the initial roadmap based on knowledge gaps."""
        print(f"--- Generating Roadmap for User {user.id} ---")
        knowledge_gaps = self.get_knowledge_gaps(user, user.learning_goal, threshold=0.6)
        concepts_to_cover = set(knowledge_gaps)
        goal_mastered = self.predict_mastery(user, user.learning_goal)
        print(f"Goal '{self.concepts[user.learning_goal].name}' mastered check: {goal_mastered}")
        if not goal_mastered:
            concepts_to_cover.add(user.learning_goal)
            queue = [user.learning_goal]; visited = {user.learning_goal}
            while queue:
                cid = queue.pop(0)
                if cid not in self.concepts: continue
                for prereq_id in self.concepts[cid].prerequisites:
                     if prereq_id in self.concepts and prereq_id not in visited:
                         visited.add(prereq_id)
                         if not self.predict_mastery(user, prereq_id):
                              concepts_to_cover.add(prereq_id)
                         queue.append(prereq_id)

        print(f"Concepts identified to cover: {[self.concepts[c].name for c in concepts_to_cover if c in self.concepts]}")
        modules_for_roadmap = {}
        for concept_id in concepts_to_cover:
            module = self._find_module_for_concept(concept_id)
            if module and module.id not in modules_for_roadmap:
                 modules_for_roadmap[module.id] = module
        user.current_roadmap = sorted(list(modules_for_roadmap.keys()))
        print(f"Roadmap for User {user.id}: Modules {[self.modules[mid].name for mid in user.current_roadmap if mid in self.modules]}")
        user.completed_modules = set()

    def predict_mastery(self, user, concept_id, threshold=0.75):
        """Predicts mastery based on the DKT model's output probability."""
        if concept_id not in self.concepts: return False
        mastery_prob = user.knowledge_state.get(concept_id, 0.0)
        return mastery_prob >= threshold

    def get_knowledge_gaps(self, user, target_concept_id, threshold=0.5):
        """Identifies concepts (and prerequisites) user likely hasn't mastered."""
        gaps = []; concepts_to_check = set(); queue = [target_concept_id]; visited = set()
        while queue:
            cid = queue.pop(0)
            if cid in visited or cid not in self.concepts: continue
            visited.add(cid); concepts_to_check.add(cid)
            for prereq_id in self.concepts[cid].prerequisites:
                if prereq_id not in visited and prereq_id in self.concepts:
                    queue.append(prereq_id)
        print(f"Checking concepts for gaps (target '{self.concepts.get(target_concept_id, Concept(target_concept_id, 'Unknown')).name}'): {[self.concepts[c].name for c in concepts_to_check if c in self.concepts]}")
        for concept_id in concepts_to_check:
             if concept_id in self.concepts and not self.predict_mastery(user, concept_id, threshold):
                 gaps.append(concept_id)
        print(f"DKT: Identified gaps for goal '{self.concepts.get(target_concept_id, Concept(target_concept_id, 'Unknown')).name}' (below {threshold}) for user {user.id}: {[self.concepts.get(g, Concept(g, 'Unknown')).name for g in gaps]}")
        return gaps

    def check_goal_completion(self, user):
        """Checks if the primary learning goal is met."""
        if user.learning_goal and user.learning_goal in self.concepts:
            is_complete = self.predict_mastery(user, user.learning_goal, threshold=0.85)
            print(f"Checking Goal Completion for User {user.id} (Goal: {user.learning_goal}): {is_complete}")
            return is_complete
        return False

# --- Sample Data (MODIFIED for Multiple Choice) ---
concepts_data = [
    Concept("C1", "What is Money?"), Concept("C2", "Saving vs Investing", ["C1"]),
    Concept("C3", "What is a Stock?", ["C1"]), Concept("C4", "What is a Mutual Fund?", ["C1", "C2"]),
    Concept("C5", "Types of Mutual Funds", ["C4"]), Concept("C6", "What is an Equity Fund?", ["C3", "C5"]),
    Concept("C7", "How to Choose Equity Funds", ["C6"])
]

questions_data = [
    # C1
    Question("Q1", "C1", "What is the primary function of money?", {'A': 'Store of Value', 'B': 'Unit of Account', 'C': 'Medium of Exchange', 'D': 'Standard of Deferred Payment'}, 'C'),
    Question("Q11", "C1", "Which is NOT typically considered one of the three main functions of money?", {'A': 'Store of Value', 'B': 'Medium of Exchange', 'C': 'Source of Credit', 'D': 'Unit of Account'}, 'C'),
    Question("Q19", "C1", "Inflation represents a decrease in:", {'A': 'The general price level', 'B': 'The purchasing power of money', 'C': 'The supply of money', 'D': 'Interest rates'}, 'B'),
    # C2
    Question("Q2", "C2", "Keeping cash under a mattress is an example of:", {'A': 'Investing', 'B': 'Saving', 'C': 'Borrowing', 'D': 'Lending'}, 'B'),
    Question("Q9", "C2", "Which generally offers higher POTENTIAL returns but also involves higher risk?", {'A': 'Saving account', 'B': 'Investing in stocks', 'C': 'Government bonds', 'D': 'Fixed deposit'}, 'B'),
    Question("Q12", "C2", "Compound interest is calculated on:", {'A': 'Principal only', 'B': 'Accumulated interest only', 'C': 'Principal and accumulated interest', 'D': 'Future deposits only'}, 'C'),
    # C3
    Question("Q3", "C3", "Buying a share of a company typically makes you a partial:", {'A': 'Creditor', 'B': 'Debtor', 'C': 'Owner', 'D': 'Manager'}, 'C'),
    Question("Q13", "C3", "A distribution of a company's profits to shareholders is called a:", {'A': 'Bond', 'B': 'Dividend', 'C': 'Premium', 'D': 'Coupon'}, 'B'),
    Question("Q14", "C3", "IPO stands for:", {'A': 'Internal Profit Option', 'B': 'Initial Public Offering', 'C': 'Investment Portfolio Outline', 'D': 'Indexed Private Obligation'}, 'B'),
    # C4
    Question("Q4", "C4", "A pool of money collected from many investors to invest in securities like stocks and bonds is called a:", {'A': 'Hedge Fund', 'B': 'Pension Fund', 'C': 'Mutual Fund', 'D': 'Savings Account'}, 'C'),
    Question("Q7", "C4", "Do mutual funds typically guarantee a return on investment?", {'A': 'Yes, always', 'B': 'Yes, if managed well', 'C': 'No, returns are subject to market risk', 'D': 'Only bond funds guarantee returns'}, 'C'),
    Question("Q15", "C4", "NAV stands for Net Asset Value. It represents the:", {'A': 'Total profit of the fund', 'B': 'Initial investment required', 'C': 'Per-share market value of the fund', 'D': 'Annual management fee'}, 'C'),
    # C5
    Question("Q5", "C5", "A mutual fund that invests primarily in stocks is called a(n):", {'A': 'Bond Fund', 'B': 'Money Market Fund', 'C': 'Balanced Fund', 'D': 'Equity Fund'}, 'D'),
    Question("Q10", "C5", "Compared to equity funds, bond funds are generally considered:", {'A': 'Higher risk, higher potential return', 'B': 'Lower risk, lower potential return', 'C': 'Lower risk, higher potential return', 'D': 'Same risk, same potential return'}, 'B'),
    Question("Q16", "C5", "A mutual fund that invests primarily in government and corporate debt is a:", {'A': 'Equity Fund', 'B': 'Bond Fund', 'C': 'Commodity Fund', 'D': 'Real Estate Fund'}, 'B'),
    # C6
    Question("Q6", "C6", "Equity funds primarily invest in:", {'A': 'Bonds', 'B': 'Real Estate', 'C': 'Commodities', 'D': 'Stocks'}, 'D'),
    Question("Q17", "C6", "A 'Large-Cap' equity fund primarily invests in companies with:", {'A': 'Small market value', 'B': 'Medium market value', 'C': 'Large market value', 'D': 'International operations only'}, 'C'),
    # C7
    Question("Q8", "C7", "An investor's willingness to accept potential losses for higher potential gains is called:", {'A': 'Investment Horizon', 'B': 'Risk Tolerance', 'C': 'Diversification', 'D': 'Liquidity Preference'}, 'B'),
    Question("Q18", "C7", "The annual fee charged by a mutual fund to cover its operational costs is the:", {'A': 'Entry Load', 'B': 'Exit Load', 'C': 'Expense Ratio', 'D': 'Brokerage Fee'}, 'C'),
    Question("Q42", "C7", "An investor saving for a goal 30 years away likely has a higher ___ than someone saving for a goal next year.", {'A': 'Need for liquidity', 'B': 'Risk tolerance', 'C': 'Expense ratio preference', 'D': 'Certainty requirement'}, 'B'),
]

modules_data = [
    Module("M1", "Intro to Finance", ["C1", "C2"], ["Money: What it is, why it matters.", "Saving Basics: Setting goals, emergency funds.", "Investing Intro: Why consider it? Saving vs Investing differences."]),
    Module("M2", "Understanding Stocks", ["C3"], ["What Owning Stock Means: You're a part-owner!", "How Stocks are Traded: Exchanges, IPOs.", "Dividends: Getting paid from profits."]),
    Module("M3", "Mutual Funds Basics", ["C4"], ["Pooling Money: The Mutual Fund Concept.", "How NAV (Net Asset Value) works.", "Advantages (Diversification, Management) and Disadvantages (Fees)."]),
    Module("M4", "Diving into Fund Types", ["C5"], ["Equity Funds: Investing in stocks.", "Bond Funds: Investing in loans (bonds).", "Comparing Risk: Equity vs. Bond Funds."]),
    Module("M5", "Focus on Equity Funds", ["C6"], ["How Equity Funds Work: Holding many stocks.", "Common Types: Large-cap (big companies), Small-cap (smaller companies)."]),
    Module("M6", "Choosing Your Equity Fund", ["C7"], ["Assessing Your Risk Tolerance: How much risk can you handle?", "Understanding Fund Objectives: What is the fund trying to achieve?", "Key Metrics: Expense Ratio (fees)."])
]
# Assign practice questions (ensure these question IDs exist and are MC)
modules_data[0].practice_questions = ["Q1", "Q2", "Q9"]
modules_data[1].practice_questions = ["Q3", "Q13", "Q14"]
modules_data[2].practice_questions = ["Q4", "Q7", "Q15"]
modules_data[3].practice_questions = ["Q5", "Q10", "Q16"]
modules_data[4].practice_questions = ["Q6", "Q17"]
modules_data[5].practice_questions = ["Q8", "Q18", "Q42"]

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
groq_api_key = os.environ.get("GROQ_API_KEY", "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") # Replace or use env var
if groq_api_key == "YOUR_GROQ_API_KEY_HERE":
    print("Warning: Replace 'YOUR_GROQ_API_KEY_HERE' with your actual Groq API key or set the GROQ_API_KEY environment variable.")

finedu_system = FinEduSystem(
    concepts=[Concept(c.id, c.name, c.prerequisites) for c in concepts_data],
    questions=[Question(q.id, q.concept_id, q.text, q.options, q.correct_option) for q in questions_data],
    modules=modules_data,
    dkt_model_class=LSTMDKTModel,
    groq_api_key=groq_api_key
)

user_sessions = {} # In-memory storage

# --- API Endpoints (Largely same structure, but data handled differently) ---

@app.route('/')
def home():
    return "FinEdu Adaptive Learning Backend (Multiple Choice) is running!"

@app.route('/start', methods=['POST'])
def start_session():
    data = request.json or {}
    name = data.get('name', 'QuizUser')
    age = data.get('age', 25)
    learning_goal_id = data.get('learning_goal', 'C7')

    if learning_goal_id not in finedu_system.concepts:
         return jsonify({"error": f"Invalid learning goal ID: {learning_goal_id}"}), 400

    user_id = str(uuid.uuid4())
    user = User(user_id, name, age)
    user_sessions[user_id] = user

    # Use a subset of questions for the initial quiz
    initial_quiz_ids = data.get('initial_quiz_ids', ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    valid_quiz_ids = [qid for qid in initial_quiz_ids if qid in finedu_system.questions and hasattr(finedu_system.questions[qid], 'options')]
    if len(valid_quiz_ids) != len(initial_quiz_ids): print(f"Warning: Some initial quiz IDs were invalid/non-MC for user {user_id}")
    if not valid_quiz_ids: return jsonify({"error": "No valid initial quiz questions found."}), 400

    first_item = finedu_system.start_session_for_user(user, valid_quiz_ids, learning_goal_id)
    return jsonify({"user_id": user_id, "next_item": first_item})

@app.route('/answer', methods=['POST'])
def handle_answer():
    data = request.json
    if not data: return jsonify({"error": "Request body must be JSON"}), 400

    user_id = data.get('user_id')
    question_id = data.get('item_id')
    item_type = data.get('item_type') # 'quiz_question' or 'practice_question'
    selected_option = data.get('answer') # Frontend sends the selected option key ('A', 'B', etc.)

    if not all([user_id, question_id, item_type, selected_option]):
        return jsonify({"error": "Missing required fields (user_id, item_id, item_type, answer with option key)"}), 400

    user = user_sessions.get(user_id)
    if not user: return jsonify({"error": "User session not found"}), 404
    if user.state not in ["QUIZ", "LEARNING"]: return jsonify({"error": f"Cannot process answer in current state: {user.state}"}), 400
    # Add more validation if needed (e.g., check expected question ID)

    result = finedu_system.process_user_answer(user, question_id, item_type, selected_option)
    if isinstance(result, tuple): return jsonify(result[0]), result[1]
    return jsonify(result)

@app.route('/get_next_item', methods=['POST'])
def get_next_item():
    data = request.json
    if not data: return jsonify({"error": "Request body must be JSON"}), 400
    user_id = data.get('user_id')
    if not user_id: return jsonify({"error": "Missing user_id"}), 400
    user = user_sessions.get(user_id)
    if not user: return jsonify({"error": "User session not found"}), 404

    # Typically called after module content
    next_item = finedu_system.get_next_item_for_user(user)
    return jsonify({"user_id": user_id, "next_item": next_item})

# Keep /get_state endpoint as before for debugging if needed
@app.route('/get_state', methods=['GET'])
def get_user_state():
    user_id = request.args.get('user_id')
    if not user_id: return jsonify({"error": "Missing user_id parameter"}), 400
    user = user_sessions.get(user_id)
    if not user: return jsonify({"error": "User session not found"}), 404
    state_data = { "user_id": user.id, "name": user.name, "state": user.state, "learning_goal": user.learning_goal, "knowledge_state": {finedu_system.concepts.get(cid, Concept(cid,"??")).name: round(prob,3) for cid, prob in user.knowledge_state.items()}, "current_roadmap": user.current_roadmap, "completed_modules": list(user.completed_modules), "current_module_id": user.current_module_id, "current_quiz_index": user.current_quiz_index, "initial_quiz_ids": user.initial_quiz_ids, "current_module_practice_index": user.current_module_practice_index, "interaction_history_count": len(user.interaction_history) }
    return jsonify(state_data)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting FinEdu Flask Server (Multiple Choice Quiz)...")
    app.run(host='0.0.0.0', port=5009, debug=True)