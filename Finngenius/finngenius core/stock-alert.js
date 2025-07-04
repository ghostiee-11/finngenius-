// --- File: stock-alert.js ---

document.addEventListener('DOMContentLoaded', function() {
    // --- Configuration ---
    // >>> IMPORTANT <<< Choose the correct API_BASE_URL based on your setup:
    // Option 1: If frontend and backend are served from the SAME domain/port
    // const API_BASE_URL = '/api';

    // Option 2: If frontend and backend are on DIFFERENT domains/ports (RECOMMENDED FOR DEV)
    // Replace 'http://localhost:5001' if your Flask server runs elsewhere.
    const API_BASE_URL = 'http://localhost:5001/api';
    // --------------------------------------------------------------------

    console.log(`Using API Base URL: ${API_BASE_URL}`);

    // --- DOM Elements ---
    const stockSelect = document.getElementById('stock-select');
    const priceContainer = document.getElementById('price-container');
    const latestPrice = document.getElementById('latest-price');
    const priceError = document.getElementById('price-error');
    const fetchError = document.getElementById('fetch-error'); // Stock list fetch error
    const targetPriceInput = document.getElementById('target-price');
    const whatsappNumberInput = document.getElementById('whatsapp-number');
    const setAlertBtn = document.getElementById('set-alert-btn');
    const alertForm = document.getElementById('alert-form'); // Get form element
    const alertFeedback = document.getElementById('alert-feedback'); // Optional feedback div

    // --- Helper Functions ---

    /** Shows the stock list fetch error message. */
    function showFetchError(message) {
        console.error("Stock List Fetch Error:", message);
        fetchError.textContent = message || 'Could not load stock list. Failed to fetch.';
        fetchError.style.display = 'block';
        // Reset select to initial error state
        stockSelect.innerHTML = '<option value="" disabled selected>-- Error Loading Stocks --</option>';
        stockSelect.disabled = true; // Disable select on error
    }
    /** Hides the stock list fetch error message. */
    function hideFetchError() {
        fetchError.style.display = 'none';
        stockSelect.disabled = false; // Re-enable select
    }

    /** Shows the price fetch error message. */
    function showPriceError(message) {
        console.error("Price Fetch Error:", message);
        priceError.textContent = message || 'Could not fetch stock price.';
        priceError.style.display = 'block';
        priceContainer.style.display = 'none'; // Hide price display
    }
    /** Hides the price fetch error message. */
    function hidePriceError() {
        priceError.style.display = 'none';
    }

     /** Displays feedback after trying to set an alert */
    function displayActionFeedback(message, isSuccess = true) {
        alertFeedback.textContent = message;
        alertFeedback.className = 'alert-status-feedback ' + (isSuccess ? 'success' : 'error');
    }
     /** Clears the alert feedback area */
    function clearActionFeedback() {
        alertFeedback.textContent = '';
        alertFeedback.className = 'alert-status-feedback'; // Hide it
    }


    // --- Core Logic ---

    /** Fetches the list of stocks and populates the dropdown. */
    async function populateStockList() {
        const url = `${API_BASE_URL}/stocks`;
        console.log(`Fetching stock list from: ${url}`);
        // Keep initial "Loading" option visible
        hideFetchError(); // Hide any previous errors
        clearActionFeedback();

        try {
            const response = await fetch(url);
            let responseData;
            const responseText = await response.text();
            try { responseData = responseText ? JSON.parse(responseText) : null; }
            catch (jsonError) {
                console.error(`Failed to parse JSON from ${url}:`, responseText);
                if (!response.ok) throw new Error(`Server error (Status: ${response.status})`);
                throw new Error("Invalid data format received.");
            }

            if (!response.ok) {
                let errorMsg = responseData?.error || `Failed to load (Status: ${response.status})`;
                if (response.status === 404) errorMsg += ` - Check backend & API URL (${API_BASE_URL}).`;
                throw new Error(errorMsg);
            }

            const stocks = responseData;
            if (!stocks || !Array.isArray(stocks)) throw new Error("Invalid stock data received.");
            if (stocks.length === 0) throw new Error("No stocks available from source.");

            // --- Populate Dropdown ---
            stockSelect.innerHTML = ''; // Clear loading/error option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = '-- Select a Stock --';
            defaultOption.disabled = true;
            defaultOption.selected = true;
            stockSelect.appendChild(defaultOption);

            let populatedCount = 0;
            stocks
                .sort((a, b) => (a.Name || a.Symbol || '').localeCompare(b.Name || b.Symbol || ''))
                .forEach(stock => {
                     if (!stock || typeof stock.Symbol !== 'string' || typeof stock.Name !== 'string' || !stock.Symbol || !stock.Name) return;
                     const option = document.createElement('option');
                     option.value = stock.Symbol.trim(); // Use Symbol as value
                     option.textContent = `${stock.Name.trim()} (${stock.Symbol.trim()})`;
                     stockSelect.appendChild(option);
                     populatedCount++;
                });

            if (populatedCount === 0) throw new Error("No valid stocks found.");

            stockSelect.disabled = false; // Enable select
            console.log(`Successfully populated ${populatedCount} stocks.`);

        } catch (error) {
            console.error("Error populating stock list:", error);
            showFetchError(`Stock list error: ${error.message}`);
            // Select remains disabled via showFetchError
        }
    }

    /** Fetches and displays the price for the selected stock. */
    async function fetchAndDisplayPrice(symbol) {
        if (!symbol) {
            priceContainer.style.display = 'none';
            hidePriceError();
            return;
        }
        const url = `${API_BASE_URL}/stock-price/${encodeURIComponent(symbol)}`;
        console.log(`Fetching price for ${symbol} from ${url}`);

        latestPrice.textContent = 'Loading...'; // Show loading state
        priceContainer.style.display = 'block';
        hidePriceError();
        clearActionFeedback();

        try {
            const response = await fetch(url);
            let responseData;
            const responseText = await response.text();
            try { responseData = responseText ? JSON.parse(responseText) : null; }
            catch (jsonError) {
                console.error(`Failed to parse JSON from ${url}:`, responseText);
                if (!response.ok) throw new Error(`Server error (Status: ${response.status})`);
                throw new Error("Invalid price data format.");
            }

            if (!response.ok) {
                 const errorMsg = responseData?.error || `Failed to fetch price (Status: ${response.status})`;
                 throw new Error(errorMsg);
            }

            if (responseData && responseData.price !== undefined) {
                const price = responseData.price;
                const currency = responseData.currency || 'INR'; // Default to INR
                latestPrice.textContent = `${currency} ${price.toFixed(2)}`;
                targetPriceInput.value = price.toFixed(2); // Pre-fill target price
                console.log(`Price updated for ${symbol}: ${price}`);
            } else {
                throw new Error("Price data missing in response.");
            }

        } catch (error) {
            console.error(`Error fetching price for ${symbol}:`, error);
            showPriceError(`Price fetch error: ${error.message}`);
            // Price container hidden by showPriceError
            targetPriceInput.value = ''; // Clear target price if fetch fails
        }
    }

    /** Handles form submission for setting an alert. */
    async function handleAlertSubmit(event) {
        event.preventDefault(); // Prevent default form submission
        clearActionFeedback();

        const selectedStock = stockSelect.value;
        const targetPrice = targetPriceInput.value;
        const whatsappNumber = whatsappNumberInput.value.trim();
        const condition = '>='; // Hardcoded

        // --- Validation ---
        if (!selectedStock) {
            displayActionFeedback('Please select a stock first.', false); // Use feedback div
            // alert('Please select a stock.'); // Old way
            return;
        }
        if (!targetPrice || isNaN(parseFloat(targetPrice)) || parseFloat(targetPrice) <= 0) {
            displayActionFeedback('Please enter a valid positive target price.', false);
            // alert('Please enter a valid positive target price.');
            targetPriceInput.focus();
            return;
        }
        // Use pattern from HTML for WhatsApp validation
        const phoneRegex = new RegExp(whatsappNumberInput.pattern);
        const phoneTitle = whatsappNumberInput.getAttribute('title') || 'Please enter a valid WhatsApp number.';
        if (!whatsappNumber || !phoneRegex.test(whatsappNumber)) {
            displayActionFeedback(phoneTitle, false);
            // alert(phoneTitle);
            whatsappNumberInput.focus();
            return;
        }

        // --- Loading State ---
        setAlertBtn.disabled = true;
        setAlertBtn.textContent = 'Setting Alert...';
        displayActionFeedback('Processing...', true); // Neutral feedback

        const payload = {
            symbol: selectedStock,
            target_price: parseFloat(targetPrice),
            whatsapp_number: whatsappNumber,
            condition: condition
        };
        const url = `${API_BASE_URL}/alerts`;
        console.log(`Sending Alert Request to: ${url}`, payload);

        // --- API Call ---
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            let result;
            const responseText = await response.text();
            try { result = responseText ? JSON.parse(responseText) : null; }
            catch (jsonError) {
                console.error(`Failed to parse JSON from ${url}:`, responseText);
                if (!response.ok) throw new Error(`Server error (Status: ${response.status})`);
                throw new Error("Invalid response format.");
            }

            if (!response.ok) {
                let errorMsg = result?.error || `Failed to set alert (Status: ${response.status})`;
                if (response.status === 400) errorMsg += ` - Check inputs.`;
                throw new Error(errorMsg);
            }

            // --- Success ---
            const successMsg = result?.message || `Alert set successfully!`;
            displayActionFeedback(`${successMsg} ${result?.alert_id ? '(ID: ...' + result.alert_id.slice(-6) + ')' : ''}`, true);
            // alertForm.reset(); // Optionally reset form fields after success
            targetPriceInput.value = ''; // Clear target price
            whatsappNumberInput.value = ''; // Clear number


        } catch (error) {
            console.error("Error setting alert:", error);
            let displayError = `Error: ${error.message}`;
             if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
                 displayError += ` - Check network/CORS/API URL (${API_BASE_URL}).`;
             }
            displayActionFeedback(displayError, false); // Use feedback div
            // alert(`Error setting alert: ${displayError}`); // Old way
        } finally {
            // --- Always Re-enable Button ---
            setAlertBtn.disabled = false;
            setAlertBtn.textContent = 'Set Alert';
        }
    }


    // --- Event Listeners ---
    stockSelect.addEventListener('change', function() {
        const selectedStockSymbol = this.value;
        fetchAndDisplayPrice(selectedStockSymbol);
    });

    alertForm.addEventListener('submit', handleAlertSubmit); // Listen on form submit

    // --- Initial Load ---
    populateStockList();

});