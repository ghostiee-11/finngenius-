// goal.js - Specific JavaScript for goal.html
// Displays overall allocation chart + individual breakdown charts per frequency

document.addEventListener('DOMContentLoaded', () => {
    // --- Constants ---
    const API_BASE_URL = 'http://127.0.0.1:5002'; // Your Investment Server Port (Ensure it's the 5-Asset version)

    // --- Element References ---
    const goalForm = document.getElementById('goal-form');
    const submitButton = document.getElementById('submit-button');
    const marketContextContent = document.getElementById('market-context-content');
    const marketLoading = document.getElementById('market-loading');
    const marketLastUpdated = document.getElementById('market-last-updated');
    const recommendationSection = document.getElementById('recommendation-section');
    const recommendationOutput = document.getElementById('recommendation-output');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');
    const displayGoalName = document.getElementById('display-goal-name');
    const displayGoalTarget = document.getElementById('display-goal-target');
    const displayGoalDuration = document.getElementById('display-goal-duration');
    const displayPortfolioReturn = document.getElementById('display-portfolio-return');
    const allocationList = document.getElementById('allocation-list'); // For overall allocation list
    const allocationChartContainer = document.getElementById('allocation-chart-container'); // For overall allocation chart
    const allocationChartCanvas = document.getElementById('allocationChart'); // Overall allocation canvas
    const breakdownPeriods = document.getElementById('breakdown-periods'); // Container for all frequency breakdowns
    const explanationText = document.getElementById('explanation-text');
    const logMonthlyInvestmentBtn = document.getElementById('log-monthly-investment-btn');
    const logStatus = document.getElementById('log-status');

    let allocationChartInstance = null; // For the main allocation chart
    let breakdownChartInstances = {}; // Store breakdown chart instances { 'monthly': chart, ... }

    // Define colors for the chart (ensure enough colors for potential assets)
    const CHART_COLORS = [
        '#6366f1', // Indigo
        '#a855f7', // Purple
        '#f59e0b', // Amber
        '#3b82f6', // Blue
        '#22c55e', // Green
        '#ec4899', // Pink
        '#ef4444', // Red
        '#84cc16', // Lime
    ];

    // --- Helper Functions ---
    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe.replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">").replace(/"/g, "").replace(/'/g, "'");
    }

    function formatCurrency(amount, currency = 'INR') {
        if (amount === null || amount === undefined || isNaN(amount)) { return 'N/A'; }
        try { return new Intl.NumberFormat('en-IN', { style: 'currency', currency: currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(amount); }
        catch (error) { console.warn("Currency formatting failed:", error); return `${currency} ${amount.toFixed(2)}`; }
    }

    function showLoading(isLoading, section = 'recommendation') {
        if (section === 'recommendation') {
            if (loadingSpinner) loadingSpinner.style.display = isLoading ? 'flex' : 'none';
            if (submitButton) submitButton.disabled = isLoading;
            if (isLoading) {
                if (recommendationOutput) recommendationOutput.style.display = 'none';
                if (errorMessage) errorMessage.style.display = 'none';
                if (recommendationSection) recommendationSection.style.display = 'block';
            }
        } else if (section === 'market') {
             if (marketLoading) marketLoading.style.display = isLoading ? 'block' : 'none';
             if (isLoading) { if (marketContextContent) marketContextContent.innerHTML = ''; }
        }
    }

    function showError(message, section = 'recommendation') {
        const errorElement = section === 'recommendation' ? errorMessage : null;
        if (errorElement) { errorElement.textContent = `Error: ${escapeHtml(message)}`; errorElement.style.display = 'block'; }
        if (section === 'recommendation') {
            if (recommendationOutput) recommendationOutput.style.display = 'none';
            if (recommendationSection) recommendationSection.style.display = 'block';
        } else if (section === 'market') {
            if (marketContextContent) marketContextContent.innerHTML = `<p class="error-message" style="text-align:center;">Could not load market data. ${escapeHtml(message)}</p>`;
        }
        console.error(`Error (${section}):`, message);
    }

    // --- Market Context ---
    // (fetchMarketContext and displayMarketContext remain the same as the previous version
    // - fetching full context, displaying only Indices, Crypto, Gold prices without trends)
    async function fetchMarketContext() {
        showLoading(true, 'market');
        if (marketLastUpdated) marketLastUpdated.textContent = 'Fetching...';
        try {
            const response = await fetch(`${API_BASE_URL}/market/context`);
            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try { const errorJson = await response.json(); errorMsg = errorJson.detail || JSON.stringify(errorJson); } catch (e) {}
                throw new Error(errorMsg);
            }
            const data = await response.json();
            displayMarketContext(data);
            if (marketLastUpdated) marketLastUpdated.textContent = new Date().toLocaleString();
        } catch (error) {
            console.error('Failed to fetch market context:', error);
            showError(error.message || "Failed to fetch market data", 'market');
            if (marketLastUpdated) marketLastUpdated.textContent = 'Error';
        } finally { showLoading(false, 'market'); }
    }

    function displayMarketContext(data) {
        if (!marketContextContent || !data || !data.assets) { console.error("Invalid market data received."); return; }
        marketContextContent.innerHTML = '';
        const categoryOrder = ["Indices", "Crypto", "Gold"]; // Categories to display in snapshot
        let categoriesDisplayed = 0;
        for (const category of categoryOrder) {
             if (!data.assets[category]) { console.warn(`Market context missing: ${category}`); continue; }
             const categoryAssets = data.assets[category];
             const categoryDiv = document.createElement('div'); categoryDiv.className = 'market-category';
             const categoryTitle = document.createElement('h4'); categoryTitle.textContent = category; categoryDiv.appendChild(categoryTitle);
             if (Object.keys(categoryAssets).length > 0) {
                 for (const symbol in categoryAssets) {
                     const asset = categoryAssets[symbol];
                     const assetItem = document.createElement('div'); assetItem.className = 'asset-item';
                     const nameSpan = document.createElement('span'); nameSpan.textContent = escapeHtml(asset.name || symbol);
                     const priceSpan = document.createElement('span'); priceSpan.className = 'asset-price';
                     priceSpan.textContent = formatCurrency(asset.current_price, asset.currency || 'INR');
                     if (asset.current_price === null || asset.current_price === undefined) { priceSpan.textContent = 'N/A'; }
                     assetItem.appendChild(nameSpan); assetItem.appendChild(priceSpan); categoryDiv.appendChild(assetItem);
                 }
             } else { const emptyMsg = document.createElement('p'); emptyMsg.innerHTML = `<small>No specific assets listed.</small>`; categoryDiv.appendChild(emptyMsg); }
             // NO Trend info displayed here
             marketContextContent.appendChild(categoryDiv); categoriesDisplayed++;
        }
        if (categoriesDisplayed === 0) { marketContextContent.innerHTML = '<p>No market data available to display.</p>'; }
    }

    // --- Goal Recommendation ---
    async function handleGoalSubmit(event) {
        event.preventDefault();
        if (!goalForm || submitButton?.disabled) return;

        showLoading(true, 'recommendation');
        // Clear previous results including breakdown charts
        if (recommendationOutput) recommendationOutput.style.display = 'none';
        if (errorMessage) errorMessage.style.display = 'none';
        if (breakdownPeriods) breakdownPeriods.innerHTML = ''; // Clear breakdown container
        if (allocationList) allocationList.innerHTML = '';
        if (explanationText) explanationText.innerHTML = '';
        // Destroy previous charts
        if (allocationChartInstance) { allocationChartInstance.destroy(); allocationChartInstance = null; }
        Object.values(breakdownChartInstances).forEach(chart => chart?.destroy()); // Destroy all breakdown charts
        breakdownChartInstances = {}; // Reset the storage object
        if (logMonthlyInvestmentBtn) logMonthlyInvestmentBtn.disabled = true;
        if (logStatus) logStatus.textContent = '';

        const formData = new FormData(goalForm);
        const requestData = { /* ... (same as before) ... */
            goal_name: formData.get('goal-name')?.trim(), risk_profile: formData.get('risk-profile'),
            goal_duration_years: parseInt(formData.get('goal-duration'), 10), goal_target_amount: parseFloat(formData.get('goal-target-amount')),
            preferred_frequency: formData.get('preferred-frequency') };

        // Validation (same as before)
        if (!requestData.goal_name) { showError("Please enter a Goal Name."); showLoading(false, 'recommendation'); return; }
        if (isNaN(requestData.goal_duration_years) || requestData.goal_duration_years <= 0) { showError("Please enter a valid Goal Duration (years > 0)."); showLoading(false, 'recommendation'); return; }
        if (isNaN(requestData.goal_target_amount) || requestData.goal_target_amount <= 0) { showError("Please enter a valid Goal Target Amount (> 0)."); showLoading(false, 'recommendation'); return; }
        if (!requestData.risk_profile || !requestData.preferred_frequency) { showError("Please select Risk Profile and Frequency."); showLoading(false, 'recommendation'); return; }

        console.log("Sending goal request:", requestData);

        try {
            const response = await fetch(`${API_BASE_URL}/goal-recommendation`, { /* ... (same as before) ... */
                method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(requestData) });

            if (!response.ok) { /* ... (error handling same as before) ... */
                let errorMsg = `Server error: ${response.status}`; try { const errorJson = await response.json(); errorMsg = errorJson.detail || JSON.stringify(errorJson); } catch (e) { errorMsg = `Server error: ${response.status} ${response.statusText}`; } throw new Error(errorMsg); }

            const recommendationData = await response.json();
            console.log("Received recommendation:", recommendationData);
            displayRecommendation(recommendationData); // Call the updated display function
            if (recommendationOutput) recommendationOutput.style.display = 'grid';

        } catch (error) { console.error('Failed to get goal recommendation:', error); showError(error.message || "Failed to fetch recommendation."); }
        finally { showLoading(false, 'recommendation'); }
    }

    // **** THIS FUNCTION IS MODIFIED to handle breakdown display ****
    function displayRecommendation(data) {
        if (!data) { showError("Received no data or invalid data structure for recommendation."); return; }

        // 1. Goal Summary (same as before)
        if (displayGoalName) displayGoalName.textContent = escapeHtml(data.goal_name || 'N/A');
        if (displayGoalTarget) displayGoalTarget.textContent = formatCurrency(data.goal_target_amount);
        if (displayGoalDuration) displayGoalDuration.textContent = data.goal_duration_years || 'N/A';
        if (displayPortfolioReturn) displayPortfolioReturn.textContent = data.estimated_portfolio_return !== null && data.estimated_portfolio_return !== undefined ? (data.estimated_portfolio_return * 100).toFixed(1) : 'N/A';

        // 2. Assign colors dynamically based on overall allocation
        const assetColors = {};
        const allocationKeys = data.allocation ? Object.keys(data.allocation) : [];
        allocationKeys.forEach((key, index) => { assetColors[key] = CHART_COLORS[index % CHART_COLORS.length]; });

        // 3. Display Overall Allocation List (same as before)
        if (allocationList) {
            allocationList.innerHTML = '';
            if (data.allocation && allocationKeys.length > 0) {
                Object.entries(data.allocation).forEach(([asset, percentage]) => { /* ... (list item creation same as before) ... */
                    const listItem = document.createElement('li'); const colorIndicator = document.createElement('span');
                    colorIndicator.style.cssText = `display:inline-block;width:12px;height:12px;background-color:${assetColors[asset] || '#ccc'};margin-right:8px;border-radius:2px;vertical-align:middle;`;
                    const assetSpan = document.createElement('span'); assetSpan.textContent = escapeHtml(asset); assetSpan.style.verticalAlign = 'middle';
                    const percentageSpan = document.createElement('span'); percentageSpan.textContent = `${(percentage * 100).toFixed(0)}%`; percentageSpan.style.verticalAlign = 'middle';
                    listItem.appendChild(colorIndicator); listItem.appendChild(assetSpan); listItem.appendChild(percentageSpan);
                    listItem.style.borderLeft = `3px solid ${assetColors[asset] || '#ccc'}`; allocationList.appendChild(listItem);
                });
            } else { allocationList.innerHTML = '<li>No allocation data available.</li>'; }
        }

        // 4. Render Overall Allocation Chart (same as before)
        if (allocationChartCanvas && data.allocation && allocationKeys.length > 0) {
            renderAllocationChart(data.allocation, assetColors); // Uses the main canvas
        } else if (allocationChartContainer) {
            allocationChartContainer.innerHTML = '<p>Chart unavailable (no allocation data).</p>';
        }

        // 5. Display Investment Breakdown (MODIFIED to show ALL periods with charts)
        if (breakdownPeriods) {
            breakdownPeriods.innerHTML = ''; // Clear previous content
            if (data.required_investment_periods && data.required_investment_periods.length > 0) {
                 let breakdownDisplayed = false;
                 // Define order for display if needed
                 const periodOrder = ['monthly', 'quarterly', 'yearly'];
                 periodOrder.forEach(periodName => {
                     const periodData = data.required_investment_periods.find(p => p.period === periodName);
                     if (periodData) {
                         // Call the updated function which now adds a chart too
                         displayBreakdownPeriod(periodData, assetColors);
                         breakdownDisplayed = true;
                     }
                 });

                 if (!breakdownDisplayed) {
                     breakdownPeriods.innerHTML = '<p>No valid investment breakdown periods found.</p>';
                 }

                 // Enable/disable log button based on monthly data availability (same logic)
                 const monthlyDataForLog = data.required_investment_periods.find(p => p.period === 'monthly');
                 if(logMonthlyInvestmentBtn && monthlyDataForLog && monthlyDataForLog.required_total_investment !== null && monthlyDataForLog.required_total_investment > 0) { logMonthlyInvestmentBtn.disabled = false; }
                 else if (logMonthlyInvestmentBtn) { logMonthlyInvestmentBtn.disabled = true; }

            } else {
                breakdownPeriods.innerHTML = '<p>Investment breakdown not available.</p>';
                if (logMonthlyInvestmentBtn) logMonthlyInvestmentBtn.disabled = true;
            }
        } else { if (logMonthlyInvestmentBtn) logMonthlyInvestmentBtn.disabled = true; }

        // 6. Display Explanation (same as before)
        if (explanationText) { /* ... (same explanation formatting as before) ... */
            let formattedExplanation = escapeHtml(data.explanation || 'No explanation provided.');
            formattedExplanation = formattedExplanation.replace(/\*\*(.*?)\*\*|__(.*?)__/g, '<strong>$1$2</strong>'); formattedExplanation = formattedExplanation.replace(/\*(.*?)\*|_(.*?)_/g, '<em>$1$2</em>');
            formattedExplanation = formattedExplanation.replace(/^[-*]\s+(.*?)(\n|$)/gm, '<li>$1</li>'); formattedExplanation = formattedExplanation.replace(/(<li>.*?<\/li>\s*)+/g, (match) => `<ul>${match}</ul>`);
            formattedExplanation = formattedExplanation.replace(/\n(?!<[/u])/g, '<br>'); explanationText.innerHTML = formattedExplanation; }
    }

    // **** THIS FUNCTION IS MODIFIED to add a chart canvas and render the chart ****
    function displayBreakdownPeriod(periodData, assetColors) {
        if (!breakdownPeriods || !periodData || !periodData.period) return; // Need period name

        const periodName = periodData.period;
        const periodDiv = document.createElement('div');
        periodDiv.className = 'period-breakdown'; // Existing class for the whole block

        const title = document.createElement('h4');
        title.textContent = `${periodName.charAt(0).toUpperCase() + periodName.slice(1)} Investment`;
        const totalAmountSpan = document.createElement('span');
        totalAmountSpan.className = 'total-amount';
        totalAmountSpan.textContent = `Total: ${formatCurrency(periodData.required_total_investment)}`;
        title.appendChild(totalAmountSpan);
        periodDiv.appendChild(title);

        // Create a container for list and chart side-by-side (or stacked on small screens)
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'breakdown-content-wrapper'; // Add class for styling (e.g., flex/grid)

        const listContainer = document.createElement('div'); // Container for the list
        listContainer.className = 'breakdown-list-container';

        const chartContainer = document.createElement('div'); // Container for the chart
        chartContainer.className = 'breakdown-chart-container';
        const canvasId = `breakdownChart-${periodName}`;
        const canvas = document.createElement('canvas');
        canvas.id = canvasId;
        canvas.width = 180; // Suggest initial size, CSS can override
        canvas.height = 180;
        chartContainer.appendChild(canvas);

        // Filter breakdown data for non-zero amounts *before* creating list/chart data
        const validBreakdown = {};
        if (periodData.breakdown) {
            for (const [asset, amount] of Object.entries(periodData.breakdown)) {
                if (amount !== null && amount > 0.001) { // Check for amount > 0
                    validBreakdown[asset] = amount;
                }
            }
        }

        // Populate the list and prepare chart data
        if (periodData.required_total_investment !== null && periodData.required_total_investment > 0 && Object.keys(validBreakdown).length > 0) {
            const list = document.createElement('ul');
            const chartLabels = [];
            const chartData = [];
            const chartColors = [];

            Object.entries(validBreakdown).forEach(([asset, amount]) => {
                // Add to list
                const listItem = document.createElement('li');
                const assetSpan = document.createElement('span');
                const colorIndicator = document.createElement('span');
                colorIndicator.style.cssText = `display:inline-block;width:10px;height:10px;background-color:${assetColors[asset] || '#ccc'};margin-right:6px;border-radius:2px;vertical-align:baseline;`;
                assetSpan.appendChild(colorIndicator); assetSpan.appendChild(document.createTextNode(escapeHtml(asset)));
                const amountSpan = document.createElement('span'); amountSpan.textContent = formatCurrency(amount);
                listItem.appendChild(assetSpan); listItem.appendChild(amountSpan); list.appendChild(listItem);

                // Add to chart data
                chartLabels.push(asset);
                chartData.push(amount);
                chartColors.push(assetColors[asset] || '#cccccc');
            });
            listContainer.appendChild(list);

            // Render the breakdown chart AFTER the canvas is added to the DOM
            // Use requestAnimationFrame to ensure rendering happens smoothly
            requestAnimationFrame(() => {
                 renderBreakdownChart(canvasId, chartLabels, chartData, chartColors);
            });

        } else {
            // Handle cases with zero investment or no breakdown
            const message = document.createElement('p');
            message.textContent = (periodData.required_total_investment === 0)
                ? 'No investment required for this period.'
                : 'Breakdown details not available or zero for this period.';
            message.style.color = 'var(--text-secondary)';
            message.style.fontSize = '0.9em';
            listContainer.appendChild(message); // Add message to list container
            chartContainer.style.display = 'none'; // Hide the chart container if no data
        }

        contentWrapper.appendChild(listContainer);
        contentWrapper.appendChild(chartContainer);
        periodDiv.appendChild(contentWrapper);
        breakdownPeriods.appendChild(periodDiv); // Append the whole period block
    }

    // **** Renders the main OVERALL allocation chart ****
    function renderAllocationChart(allocationData, assetColors) {
        if (!allocationChartCanvas || !window.Chart) { console.error("Main chart canvas or Chart.js library not found."); return; }
        if (allocationChartInstance) { allocationChartInstance.destroy(); allocationChartInstance = null; } // Destroy previous

        const labels = Object.keys(allocationData);
        const data = Object.values(allocationData).map(val => (val * 100).toFixed(1)); // Percentages
        const backgroundColors = labels.map(label => assetColors[label] || '#cccccc');

        try {
            const ctx = allocationChartCanvas.getContext('2d');
            allocationChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: { labels: labels, datasets: [{ label: 'Allocation %', data: data, backgroundColor: backgroundColors, borderColor: 'var(--bg-card)', borderWidth: 2, hoverOffset: 4 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(context) { return `${context.label || ''}: ${context.parsed?.toFixed(1) || 0}%`; } }, /* tooltip styles */ backgroundColor:'var(--bg-secondary)',titleColor:'var(--text-primary)',bodyColor:'var(--text-secondary)',borderColor:'var(--border)',borderWidth:1 } }, cutout: '65%' }
            }); console.log("Main allocation chart rendered.");
        } catch (error) { console.error("Error rendering main chart:", error); if (allocationChartContainer) { allocationChartContainer.innerHTML = '<p>Error rendering chart.</p>'; } }
    }

    // **** NEW FUNCTION: Renders the breakdown chart for a specific period ****
    function renderBreakdownChart(canvasId, labels, data, backgroundColors) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !window.Chart) { console.error(`Breakdown chart canvas #${canvasId} or Chart.js not found.`); return; }

        const periodName = canvasId.split('-')[1]; // Extract period name ('monthly', etc.)

        // Destroy previous chart for this specific canvas ID if it exists
        if (breakdownChartInstances[periodName]) {
             breakdownChartInstances[periodName].destroy();
        }

        try {
            const ctx = canvas.getContext('2d');
            breakdownChartInstances[periodName] = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Amount (INR)', // Label for the dataset
                        data: data, // Monetary amounts
                        backgroundColor: backgroundColors,
                        borderColor: 'var(--bg-secondary)', // Match breakdown bg
                        borderWidth: 1,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false, // Allow resizing
                    plugins: {
                        legend: { display: false }, // Legend handled by list
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    // Format tooltip to show currency amount
                                    const label = context.label || '';
                                    const value = context.parsed;
                                    return `${label}: ${formatCurrency(value)}`;
                                }
                            },
                            // Tooltip styling (can customize further)
                            backgroundColor: 'var(--bg-secondary)', titleColor: 'var(--text-primary)',
                            bodyColor: 'var(--text-secondary)', borderColor: 'var(--border)', borderWidth: 1
                        }
                    },
                    cutout: '60%' // Slightly smaller cutout? Optional.
                }
            });
            console.log(`Breakdown chart #${canvasId} rendered.`);
        } catch (error) {
            console.error(`Error rendering breakdown chart #${canvasId}:`, error);
            const container = canvas.parentElement;
            if (container) container.innerHTML = '<p><small>Chart error.</small></p>';
        }
    }


    // --- Simulated Log Investment ---
    function handleLogInvestment() {
        if (!logMonthlyInvestmentBtn || !logStatus) return;
        logMonthlyInvestmentBtn.disabled = true;
        logStatus.textContent = 'Logging investment...';
        logStatus.className = 'log-status-message';
        setTimeout(() => {
            logStatus.textContent = 'Monthly investment logged (Simulated).';
            logStatus.classList.add('success');
        }, 1000);
    }


    // --- Event Listeners ---
    if (goalForm) { goalForm.addEventListener('submit', handleGoalSubmit); }
    else { console.error("Goal form not found!"); }
    if (logMonthlyInvestmentBtn) { logMonthlyInvestmentBtn.addEventListener('click', handleLogInvestment); }

    // --- Initial Load ---
    fetchMarketContext(); // Fetch context on load (will be displayed filtered)
    if (recommendationSection) recommendationSection.style.display = 'none'; // Hide recommendation initially
    if (logMonthlyInvestmentBtn) logMonthlyInvestmentBtn.disabled = true; // Ensure log button is disabled initially

}); // End DOMContentLoaded