// static/js/script.js (Final Version with Hardcoded Min Ages & All Features)

document.addEventListener('DOMContentLoaded', () => { // Ensure DOM is ready

    // --- Get DOM Elements ---
    const loanForm = document.getElementById('loan-form');
    const resultsSection = document.getElementById('results-section');
    const resultsList = document.getElementById('results-list');
    const submitButton = document.getElementById('submit-button');
    const buttonText = submitButton.querySelector('.button-text'); // Get the text part of the button
    const spinner = submitButton.querySelector('.spinner'); // Get the spinner element
    const compareButton = document.getElementById('compare-button');
    const modal = document.getElementById('comparison-modal');
    const closeModalButton = modal.querySelector('.close-button');
    const comparisonTableContainer = document.getElementById('comparison-table-container');
    const loanTypeDropdown = document.getElementById('loan-type');
    const minAgeInfoDisplay = document.getElementById('min-age-info');

    let currentResults = []; // Store the latest results for comparison

    // --- Hardcoded Minimum Age Mapping ---
    // IMPORTANT: Review these values against your actual data or requirements.
    // Keys MUST exactly match the 'value' attributes in the HTML <select> options.
    const hardcodedMinAges = {
        "Home Loan": 18,
        "Auto Loan": 21,
        "Education Loan": 18,
        "Personal Loan": 21,
        "Business Loan": 21,
        "Gold Loan": 18,
        "Loan Against Property": 21,
        "Credit Card Loan": 21,
        "Short-term Loan": 21,
        "Agricultural Loan": 18,
        "Mortgage Loan": 21,
        "Bridge Loan": 23,
        "Overdraft Facility": 23,
        "Professional Loan": 25,
        "Construction Loan": 23,
        "Loan Against Shares/Mutual Funds": 23,
        "Reverse Mortgage Loan for Senior Citizens": 60, // Renamed value in HTML, match here
        "Microfinance Loan": 18,
        "Development Loan": 21,
        "vacation loan": 21
        // Add any other loan types present in your dropdown here
    };


    // --- Event Listener for Loan Type Change ---
    loanTypeDropdown.addEventListener('change', function() {
        const selectedType = this.value; // Get the selected loan type string
        updateMinAgeDisplay(selectedType);
    });

    // Function to update the minimum age display using the HARDCODED map
    function updateMinAgeDisplay(loanType) {
         // Use hasOwnProperty for safer check if the key exists in the map
         if (loanType && hardcodedMinAges.hasOwnProperty(loanType)) {
            const minAge = hardcodedMinAges[loanType];
            minAgeInfoDisplay.textContent = `ℹ️ Minimum age typically required: ${minAge}`;
            minAgeInfoDisplay.style.display = 'block'; // Make the info div visible
        } else {
            minAgeInfoDisplay.textContent = ''; // Clear text if no type selected or not in map
            minAgeInfoDisplay.style.display = 'none'; // Hide the info div
        }
    }

    // Initialize display on page load in case a value is pre-selected
    updateMinAgeDisplay(loanTypeDropdown.value);


    // --- Form Submission Logic ---
    loanForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default page reload
        clearComparison(); // Clear previous comparison selections

        // Get form values
        const loanType = loanTypeDropdown.value;
        const loanAmount = document.getElementById('loan_amount').value;
        const loanTerm = document.getElementById('loan_term').value;
        const age = document.getElementById('age').value;
        const score = document.getElementById('score').value;

        // Basic validation
        if (!loanType || !loanAmount || !loanTerm || !age || !score) {
            alert('Please fill in all fields, including Loan Amount and Term.');
            return;
        }
        if (parseFloat(loanAmount) <= 0 || parseInt(loanTerm) <= 0) {
            alert('Please enter a valid Loan Amount and Term (greater than 0).');
            return;
        }

        // --- Start Loading State ---
        submitButton.disabled = true;
        submitButton.classList.add('loading'); // Add class to handle visuals (spinner/text hiding) via CSS
        resultsSection.style.display = 'none'; // Hide previous results section
        resultsList.innerHTML = '<p>Searching for the best loan options...</p>'; // Show temporary text

        // Prepare data payload for the API request
        const requestData = {
            loan_type: loanType,
            loan_amount: parseFloat(loanAmount),
            loan_term: parseInt(loanTerm),
            age: parseInt(age),
            score: parseInt(score)
        };

        // --- API Call ---
        const apiUrl = 'http://127.0.0.1:5010/recommend'; // Ensure this URL points to your running Flask backend
        fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData), // Convert JS object to JSON string
        })
        .then(response => {
             // Check if the HTTP response status code indicates success
            if (!response.ok) {
                // If not okay, try to parse the error message from the backend response
                return response.json().then(err => {
                    // Throw an error using the message from the backend if available
                    throw new Error(err.error || `Server Error: ${response.status}`);
                }).catch(() => {
                    // Fallback if the response body isn't JSON or doesn't contain 'error'
                    throw new Error(`Server responded with status: ${response.status}`);
                });
            }
            // If response is okay, parse the JSON body containing the loan data
            return response.json();
        })
        .then(data => {
            currentResults = data; // Store the received loan data globally for comparison feature
            displayResults(data); // Call function to render the results on the page
        })
        .catch(error => {
            // Handle network errors or errors thrown during response processing
            console.error('Error fetching recommendations:', error);
            resultsList.innerHTML = `<p class="error">An error occurred: ${error.message}. Please check the API server or network connection.</p>`;
            resultsSection.style.display = 'block'; // Show the results section to display the error
        })
        .finally(() => {
            // --- End Loading State ---
            // This block executes regardless of success or failure of the fetch call
            submitButton.disabled = false;
            submitButton.classList.remove('loading'); // Remove loading class to restore button appearance
        });
    });

    // --- Display Results Logic ---
    function displayResults(loans) {
        resultsList.innerHTML = ''; // Clear placeholder or previous results

        // Handle cases where the backend returned an error object or no data
        if (!loans || loans.error) {
            resultsList.innerHTML = `<p class="error">${loans?.error || 'Failed to load recommendations.'}</p>`;
             resultsSection.style.display = 'block';
            return;
        }

        // Handle case where the search yielded no matching loans
        if (loans.length === 0) {
            resultsList.innerHTML = '<p class="no-results">No suitable loan schemes found matching your criteria.</p>';
            resultsSection.style.display = 'block';
            return;
        }

        // Loop through each recommended loan and create an HTML card for it
        loans.forEach((loan, index) => {
            const card = document.createElement('div');
            card.classList.add('loan-card'); // Apply card styling
            card.dataset.loanId = loan.id; // Store unique loan ID for comparison tracking

            // Prepare display values, handling potential nulls or N/A cases
            const estimatedRateUsed = loan.rate_used_for_calc ? `(using ~${loan.rate_used_for_calc.toFixed(2)}%)` : '(rate info unclear for calc)';
            const emiDisplay = loan.estimated_emi > 0 ? `₹ ${loan.estimated_emi.toLocaleString('en-IN')}` : 'N/A';
            const totalInterestDisplay = loan.estimated_total_interest > 0 ? `₹ ${loan.estimated_total_interest.toLocaleString('en-IN')}` : 'N/A';
            const totalRepaymentDisplay = loan.estimated_total_repayment > 0 ? `₹ ${loan.estimated_total_repayment.toLocaleString('en-IN')}` : 'N/A';
            const minScoreDisplay = loan['Min Credit Score'] === 0 ? 'None specified' : loan['Min Credit Score']; // Handle 0 score case

            // Construct the inner HTML of the card using template literals
            card.innerHTML = `
                <h3>
                    ${loan['Loan Scheme']}
                    <span class="bank-name">(${loan['Bank Name']})</span>
                </h3>
                <p><strong>Interest Rate:</strong> <span>${loan['Interest Rate']}</span></p>
                <p><strong>Min. Credit Score:</strong> <span>${minScoreDisplay}</span></p>
                <p><strong>Min. Age:</strong> <span>${loan['Minimum Age']}</span></p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 10px 0;">
                <p><strong>Est. Monthly EMI:</strong> <span>${emiDisplay} <small>${estimatedRateUsed}</small></span></p>
                <p><strong>Est. Total Interest:</strong> <span>${totalInterestDisplay}</span></p>
                <p><strong>Est. Total Repayment:</strong> <span>${totalRepaymentDisplay}</span></p>

                <details class="eligibility-details">
                    <summary>Eligibility Criteria</summary>
                    <p>${loan['Eligibility Criteria'] || 'Not specified'}</p>
                </details>
                <div class="card-actions">
                    <label>
                        <input type="checkbox" class="compare-checkbox" data-loan-id="${loan.id}"> Compare
                    </label>
                </div>
            `;
            resultsList.appendChild(card); // Add the newly created card to the results list container
        });

        resultsSection.style.display = 'block'; // Make the results section visible
        updateCompareButtonState(); // Initialize the state of the compare button
    }

    // --- Comparison Logic ---
    // Add event listener to the results list container (event delegation)
    resultsList.addEventListener('change', (event) => {
        // Check if the changed element was a compare checkbox
        if (event.target.classList.contains('compare-checkbox')) {
            updateCompareButtonState(); // Update button state whenever a checkbox changes
        }
    });

    // Helper function to get IDs of currently selected loans for comparison
    function getSelectedLoanIds() {
        const checkboxes = resultsList.querySelectorAll('.compare-checkbox:checked');
        // Convert NodeList to Array and extract loan IDs from data attribute
        return Array.from(checkboxes).map(cb => cb.dataset.loanId);
    }

    // Updates the text and disabled state of the "Compare Selected" button
    function updateCompareButtonState() {
        const selectedIds = getSelectedLoanIds();
        const count = selectedIds.length;
        compareButton.textContent = `Compare Selected (${count})`; // Update button text
        // Enable button only when exactly 2 or 3 loans are selected
        compareButton.disabled = count < 2 || count > 3;
    }

    // Function to uncheck all comparison checkboxes
    function clearComparison() {
         const checkboxes = resultsList.querySelectorAll('.compare-checkbox');
         checkboxes.forEach(cb => cb.checked = false);
         updateCompareButtonState(); // Reset compare button state
    }

    // Event listener for the "Compare Selected" button
    compareButton.addEventListener('click', () => {
        const selectedIds = getSelectedLoanIds();
        // Proceed only if 2 or 3 loans are selected
        if (selectedIds.length >= 2 && selectedIds.length <= 3) {
            // Filter the globally stored `currentResults` to get the full data for selected loans
            const loansToCompare = currentResults.filter(loan => selectedIds.includes(loan.id));
            generateComparisonTable(loansToCompare); // Create the comparison table HTML
            modal.style.display = 'block'; // Show the comparison modal popup
        }
    });

    // Function to generate the HTML table for loan comparison
    function generateComparisonTable(loans) {
        if (!loans || loans.length === 0) {
            comparisonTableContainer.innerHTML = '<p>Error generating comparison data.</p>';
            return;
        }

        // Start building the table HTML string
        let tableHTML = '<table><thead><tr><th>Feature</th>';
        // Add table headers for each selected loan
        loans.forEach(loan => {
            tableHTML += `<th>${loan['Loan Scheme']} (${loan['Bank Name']})</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        // Define the features (rows) to include in the comparison table
        const features = [
            { key: 'Interest Rate', label: 'Interest Rate' },
            { key: 'estimated_emi', label: 'Est. Monthly EMI', format: 'currency' },
            { key: 'estimated_total_interest', label: 'Est. Total Interest', format: 'currency' },
            { key: 'estimated_total_repayment', label: 'Est. Total Repayment', format: 'currency' },
            { key: 'Min Credit Score', label: 'Min. Credit Score', format: 'score' },
            { key: 'Minimum Age', label: 'Min. Age' },
            { key: 'Eligibility Criteria', label: 'Eligibility Notes', format: 'text' },
        ];

        // Loop through each feature to create table rows
        features.forEach(feature => {
            tableHTML += `<tr><th>${feature.label}</th>`; // Add row header (feature name)
            // Loop through each loan to get its value for the current feature
            loans.forEach(loan => {
                let value = loan[feature.key]; // Get the raw value

                // Apply formatting based on the feature type
                if (feature.format === 'currency' && typeof value === 'number' && value > 0) {
                    // Format currency with commas and no decimals
                    value = `₹ ${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
                } else if (feature.format === 'currency') {
                    value = 'N/A'; // If value is not a positive number
                } else if (feature.format === 'score' && value === 0) {
                    value = 'None specified'; // Special text for 0 score
                } else if (feature.format === 'text') {
                    // Wrap long text in small tags for potentially smaller font
                    value = `<small>${value || 'Not specified'}</small>`;
                } else if (value === null || value === undefined || value === '' || (typeof value === 'number' && value <= 0 && feature.key.includes('estimated'))) {
                     // General catch-all for missing or non-positive estimated values
                     value = 'N/A';
                 }
                // Add the formatted value as a table data cell
                tableHTML += `<td>${value}</td>`;
            });
            tableHTML += '</tr>'; // Close the table row
        });

        tableHTML += '</tbody></table>'; // Close the table body and table
        comparisonTableContainer.innerHTML = tableHTML; // Inject the generated table into the modal
    }

    // --- Modal Close Logic ---
    // Close modal when the 'X' button is clicked
    closeModalButton.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close modal if user clicks anywhere outside the modal content area
    window.addEventListener('click', (event) => {
        if (event.target === modal) { // Check if the click target is the modal background itself
            modal.style.display = 'none';
        }
    });

}); 