// --- scam-checker.js ---
(function() { // IIFE to avoid polluting global scope

    // --- Configuration ---
    // Make sure this URL and PORT match your running Flask backend (app.py)
    const API_URL = 'http://localhost:5001/api/check';
    const SCAM_THRESHOLD = 65; // Probability % threshold to classify as scam (adjust if needed)

    // --- DOM Elements ---
    let messageInput, checkButton, resultContainer, probabilityResultSpan,
        verdictResultSpan, linksResultContainer, suspiciousLinksList,
        errorMessageContainer, loadingIndicator;

    // --- Initialization ---
    document.addEventListener('DOMContentLoaded', () => {
        // Get references to all needed elements
        messageInput = document.getElementById('messageInput');
        checkButton = document.getElementById('checkButton');
        resultContainer = document.getElementById('resultContainer');
        probabilityResultSpan = document.getElementById('probabilityResult')?.querySelector('.value'); // Use optional chaining
        verdictResultSpan = document.getElementById('verdictResult')?.querySelector('.value');
        linksResultContainer = document.getElementById('linksResult');
        suspiciousLinksList = document.getElementById('suspiciousLinksList');
        errorMessageContainer = document.getElementById('errorMessage');
        loadingIndicator = document.getElementById('loadingIndicator');

        // Basic check if elements exist
        if (!messageInput || !checkButton || !resultContainer || !probabilityResultSpan || !verdictResultSpan || !linksResultContainer || !suspiciousLinksList || !errorMessageContainer || !loadingIndicator) {
            console.error("Scam Checker Error: One or more required HTML elements not found!");
            // Optionally display an error to the user on the page
            const body = document.querySelector('body');
            if (body) body.insertAdjacentHTML('afterbegin', '<p style="color: red; text-align: center; padding: 1rem;">Error: Page elements missing. Cannot initialize scam checker.</p>');
            return; // Stop initialization if elements are missing
        }

        // Add event listeners
        checkButton.addEventListener('click', handleCheck);
        messageInput.addEventListener('keypress', (e) => {
            // Trigger check on Enter key (but not Shift+Enter)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent newline in textarea
                handleCheck();
            }
        });

        // Initialize Lucide icons if the library is present
        if (typeof lucide !== 'undefined' && lucide.createIcons) {
           try { lucide.createIcons(); } catch(e) { console.error("Lucide init error:", e); }
        }
        console.log("Scam Checker Initialized.");
    });

    // --- Event Handler ---
    async function handleCheck() {
        const message = messageInput.value.trim();
        if (!message) {
            showError("Please enter a message in the text area first.");
            return;
        }

        // Reset UI state
        hideError();
        resultContainer.style.display = 'none';
        loadingIndicator.style.display = 'block';
        checkButton.disabled = true;
        checkButton.textContent = 'Analyzing...'; // Provide feedback

        try {
            console.log(`Sending message to API: ${API_URL}`);
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json' // Explicitly accept JSON
                },
                body: JSON.stringify({ message: message }), // Ensure correct JSON format
            });

            console.log(`API Response Status: ${response.status}`);

            // Try to parse JSON regardless of status to get potential error message
            let resultJson = null;
            try {
                resultJson = await response.json();
                console.log("API Raw Response Body:", resultJson);
            } catch (jsonError) {
                 // If parsing fails even on non-ok response, that's an issue
                 if (!response.ok) {
                     // Throw generic HTTP error if JSON parsing fails for an error response
                     throw new Error(`HTTP error! Status: ${response.status}. Failed to parse error response.`);
                 }
                 // If response was OK but JSON parsing failed, treat as internal error
                 console.error("Failed to parse JSON response:", jsonError);
                 throw new Error("Received invalid response format from server.");
            }


            if (!response.ok) {
                // Use error message from JSON if available, otherwise use HTTP status
                const errorDetail = resultJson?.error || `HTTP error! Status: ${response.status}`;
                throw new Error(`Analysis failed: ${errorDetail}`);
            }

            // Proceed with displaying results if response is OK and JSON parsed
            displayResult(resultJson);

        } catch (error) {
            console.error('Error during scam check API call:', error);
            // Display the caught error message to the user
            showError(`Error: ${error.message}. Please check the console or try again later.`);
        } finally {
            // Always restore UI state
            loadingIndicator.style.display = 'none';
            checkButton.disabled = false;
            checkButton.innerHTML = `Check Message <i data-lucide="scan-search" style="margin-left: 0.5rem; width: 1.1rem; height: 1.1rem; vertical-align: middle;"></i>`; // Restore button text/icon
            // Re-render icon if needed
            if (typeof lucide !== 'undefined' && lucide.createIcons) {
                 try { lucide.createIcons(); } catch(e) { /* ignore */ }
            }
        }
    }

    // --- UI Update Functions ---
    function displayResult(result) {
        // Use default values if properties are missing in the response
        const probability = result.scam_probability ?? 0;
        // Recalculate verdict based on threshold if is_scam is missing
        const isScam = result.is_scam ?? (probability > SCAM_THRESHOLD);
        const suspiciousLinks = result.suspicious_links ?? [];

        probabilityResultSpan.textContent = `${probability}%`;
        verdictResultSpan.textContent = isScam ? "Likely Scam" : "Likely Not Scam";

        // Apply styling classes based on results
        verdictResultSpan.className = isScam ? 'value scam' : 'value not-scam';
        probabilityResultSpan.parentElement.className = getProbabilityClass(probability); // Apply class to parent div


        // Display suspicious links if any found
        if (suspiciousLinks.length > 0) {
            suspiciousLinksList.innerHTML = ''; // Clear previous links
            suspiciousLinks.forEach(link => {
                const li = document.createElement('li');
                // Sanitize link text before displaying (basic example)
                li.textContent = link.replace(/</g, "<").replace(/>/g, ">");
                suspiciousLinksList.appendChild(li);
            });
            linksResultContainer.style.display = 'block'; // Show the links section
        } else {
            linksResultContainer.style.display = 'none'; // Hide if no links
        }

        resultContainer.style.display = 'block'; // Show the main results area
    }

    function showError(message) {
        if (errorMessageContainer) {
            errorMessageContainer.textContent = message;
            errorMessageContainer.style.display = 'block';
        }
        if (resultContainer) {
            resultContainer.style.display = 'none'; // Hide results area on error
        }
        console.error("Displayed Error:", message);
    }

    function hideError() {
        if (errorMessageContainer) {
            errorMessageContainer.style.display = 'none';
            errorMessageContainer.textContent = '';
        }
    }

    function getProbabilityClass(prob) {
        // Returns CSS class based on probability for styling
        if (prob > 80) return 'result-item probability high'; // Use 80 as high threshold
        if (prob > 50) return 'result-item probability medium'; // Use 50 as medium threshold
        return 'result-item probability low';
    }

})(); // End IIFE