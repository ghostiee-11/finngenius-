# app.py (Additions highlighted)

import pandas as pd
from flask import Flask, request, jsonify, render_template, json # Import json for safe embedding
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "*"}}) # Adjust origins as needed

# --- Calculation Functions (Keep calculate_emi, estimate_rate_for_calc) ---
# ... (existing functions) ...
def calculate_emi(principal, annual_rate_percent, term_years):
    if principal <= 0 or annual_rate_percent <= 0 or term_years <= 0: return 0
    try:
        monthly_rate = (annual_rate_percent / 100) / 12; term_months = term_years * 12
        emi = principal * monthly_rate * (pow(1 + monthly_rate, term_months)) / (pow(1 + monthly_rate, term_months) - 1)
        return round(emi, 2)
    except (OverflowError, ValueError): return 0

def estimate_rate_for_calc(rate_str):
    if pd.isna(rate_str) or not isinstance(rate_str, str): return None
    rate_str = rate_str.strip().replace('%', '')
    try:
        if '-' in rate_str: low, high = map(float, rate_str.split('-')); return (low + high) / 2
        elif ' onwards' in rate_str: return float(rate_str.replace(' onwards', '').strip())
        elif '–' in rate_str: low, high = map(float, rate_str.split('–')); return (low + high) / 2
        else: return float(rate_str)
    except ValueError: return None
# --- Load Preprocessed Data ---
PICKLE_FILE = '/Users/amankumar/Desktop/ff/frontend/processed_loan_data.pkl'
df_cleaned = pd.DataFrame()
loan_types = []
loan_type_min_ages = {} # *** NEW: Initialize empty dict for age mapping ***

try:
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            processed_data = pickle.load(f)
            df_cleaned = processed_data.get('data', pd.DataFrame()) # Use .get for safety
            loan_types = processed_data.get('loan_types', [])
            loan_type_min_ages = processed_data.get('loan_type_min_ages', {}) # *** NEW: Load the age mapping ***
            print(f"Loaded preprocessed data: {len(df_cleaned)} records, {len(loan_types)} types.")
            if loan_type_min_ages:
                 print("Loaded minimum age mapping for loan types.")
    else:
        print(f"ERROR: {PICKLE_FILE} not found.")
except Exception as e:
    print(f"Error loading data: {e}")

# --- Flask Routes ---
@app.route('/')
def index_page(): # Renamed route function slightly for clarity
    """
    Renders the main HTML page and passes necessary data.
    This route is crucial if serving HTML from Flask.
    If frontend is separate, this data needs another delivery mechanism (e.g., dedicated API endpoint).
    """
    # *** NEW: Pass the age mapping as JSON to the template ***
    return render_template(
        'index.html',
        loan_types_json=json.dumps(loan_types), # Pass types as JSON too if needed by JS
        loan_type_min_ages_json=json.dumps(loan_type_min_ages) # Pass age map as JSON
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    # ... (Keep the existing recommendation logic exactly the same) ...
    # It doesn't need the loan_type_min_ages data itself.
    if df_cleaned.empty: return jsonify({"error": "Loan data is not available."}), 500
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Invalid input"}), 400
        user_age = int(data.get('age')); user_score = int(data.get('score'))
        loan_type = data.get('loan_type'); loan_amount = float(data.get('loan_amount', 0))
        loan_term = int(data.get('loan_term', 0))
        if not all([isinstance(user_age, int), isinstance(user_score, int), loan_type, isinstance(loan_amount, (int, float)), isinstance(loan_term, int)]): return jsonify({"error": "Missing or invalid parameters"}), 400
        if loan_amount <= 0 or loan_term <= 0: return jsonify({"error": "Please provide a valid Loan Amount and Term (in years) for calculations."}), 400

        filtered_df = df_cleaned[df_cleaned['Loan Type'].str.strip().str.lower() == loan_type.strip().lower()].copy()
        eligible_loans = filtered_df[(filtered_df['Minimum Age'] <= user_age) & (filtered_df['Min Credit Score'] <= user_score)].copy()
        recommended_loans = eligible_loans.sort_values(by='Min Interest Rate', ascending=True)
        results = []
        for index, loan in recommended_loans.head(10).iterrows():
            loan_dict = loan.to_dict(); loan_dict['id'] = f"loan_{index}"
            estimated_rate = estimate_rate_for_calc(loan_dict['Interest Rate']); emi = 0; total_interest = 0; total_repayment = 0
            if estimated_rate is not None and loan_amount > 0 and loan_term > 0:
                emi = calculate_emi(loan_amount, estimated_rate, loan_term)
                if emi > 0: total_repayment = round(emi * loan_term * 12, 2); total_interest = round(total_repayment - loan_amount, 2)
            loan_dict['estimated_emi'] = emi; loan_dict['estimated_total_interest'] = total_interest
            loan_dict['estimated_total_repayment'] = total_repayment; loan_dict['rate_used_for_calc'] = estimated_rate
            for key, value in loan_dict.items():
                if isinstance(value, (np.int64, np.int32)): loan_dict[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)): loan_dict[key] = float(value)
                elif pd.isna(value): loan_dict[key] = None
            results.append(loan_dict)
        return jsonify(results)
    except ValueError: return jsonify({"error": "Invalid number format for Age, Score, Amount, or Term."}), 400
    except Exception as e: print(f"Error during recommendation: {e}"); import traceback; traceback.print_exc(); return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == '__main__':
    if df_cleaned.empty and not os.path.exists(PICKLE_FILE):
         print("Cannot start: Preprocessed data missing.")
    else:
        print(f"Starting Flask server on http://0.0.0.0:5010")
        # Remember to restart the server after changing the code
        app.run(host='0.0.0.0', port=5010, debug=True)