document.addEventListener('DOMContentLoaded', function() {
    // Financial tools data - Forms and calculate functions are removed
    const financialTools = [
        { id: 'sip', title: 'SIP Calculator', description: 'Estimate returns on your SIP investment', icon: '💰🌱' },
        { id: 'compound-interest', title: 'Compound Interest Calculator', description: 'Discover the Power of Compounding Now', icon: '📈💰' },
        { id: 'savings', title: 'Savings Calculator', description: 'Estimate interest earned on savings account and total earnings', icon: '💵📊' },
        { id: 'income-tax', title: 'Income Tax Calculator', description: 'Calculate your income tax for the year', icon: '💰📊' },
        { id: 'pension', title: 'Pension Calculator', description: 'Tool to know how much you need to save for retirement', icon: '👴💰' },
        // { id: 'nps', title: 'NPS Calculator', description: 'Estimate returns on your NPS contribution', icon: '📝💰' },
        { id: 'ulip', title: 'ULIP Calculator', description: 'Helps compare plans and estimate future maturity amounts', icon: '📉💰' },
        { id: 'hlv', title: 'Human Life Value Calculator', description: 'Easily estimate your life cover based on income and future goals', icon: '👤🛡️' },
        // { id: 'save-regularly', title: 'Save Regularly', description: 'How much should I save regularly?', icon: '🐖💰' },
        // { id: 'actual-savings', title: 'Actual Savings', description: 'What will be my final savings amount?', icon: '💰✨' },
        { id: 'health-insurance', title: 'Health Insurance Premium Calculator', description: 'How can I get the best quotes for my health insurance?', icon: '⚕️🛡️' },
        { id: 'car-insurance', title: 'Car Insurance Calculator', description: 'How can I save big on car insurance?', icon: '🚗🛡️' },
        { id: 'bike-insurance', title: 'Bike Insurance Calculator', description: 'How can I find the right insurance cover for my bike?', icon: '🏍️🛡️' },
        // { id: 'life-insurance', title: 'Life Insurance Calculator', description: 'Calculate your life insurance premium easily and without hassle', icon: '👪🛡️' },
        // { id: 'term-insurance', title: 'Term Insurance Calculator', description: 'Calculate the premium of your term insurance plan in seconds!', icon: '📄🛡️' },
        // { id: 'lic', title: 'LIC Calculator', description: 'Calculate your LIC premium rate and maturity amount in a few clicks!', icon: '🛡️💼' },
        { id: 'fd', title: 'FD Calculator', description: 'How can I earn maximum interest through FD?', icon: '📝💵' },
        { id: 'investment', title: 'Investment Calculator', description: 'What will be the returns on my investment?', icon: '💰📈' },
        { id: 'gst', title: 'GST Calculator', description: 'How should I evaluate the value of goods and services?', icon: '🧾✅' },
        { id: 'hra', title: 'HRA Calculator', description: 'How can I save money on tax?', icon: '🏠💰' },
        { id: 'gratuity', title: 'Gratuity Calculator', description: 'How much gratuity will I receive when I retire?', icon: '💼💰' },
        { id: 'ppf', title: 'PPF Calculator', description: 'What will be my PPF maturity value and interest earned?', icon: '📊💰' },
        { id: 'travel-insurance', title: 'Travel Insurance Calculator', description: 'How much should I invest in Travel Insurance?', icon: '✈️🛡️' }
    ];

    const toolsGrid = document.getElementById('toolsGrid');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    // Modal elements are no longer needed for this functionality

    // Function to render tools
    function renderTools(tools) {
        toolsGrid.innerHTML = ''; // Clear existing tools

        if (tools.length === 0) {
            toolsGrid.innerHTML = '<div class="no-results">No tools found matching your search.</div>';
            return;
        }

        tools.forEach(tool => {
            const toolCard = document.createElement('div');
            toolCard.className = 'tool-card';
            // Store the tool ID to know which file to link to
            toolCard.setAttribute('data-id', tool.id);

            // Consistent button text for all cards
            const buttonText = 'Open Calculator';

            toolCard.innerHTML = `
                <div class="tool-icon">${tool.icon}</div>
                <div class="tool-title">${tool.title}</div>
                <div class="tool-description">${tool.description}</div>
                <button class="tool-button">${buttonText}</button>
            `;

            // Add event listener directly to the card for redirection
            toolCard.addEventListener('click', function() {
                const toolId = this.getAttribute('data-id');
                redirectToCalculatorPage(toolId);
            });

            toolsGrid.appendChild(toolCard);
        });
    }

    // Function to filter tools based on search query
    function filterTools(query) {
        if (!query) {
            return financialTools; // Return all tools if query is empty
        }
        query = query.toLowerCase();
        return financialTools.filter(tool =>
            tool.title.toLowerCase().includes(query) ||
            tool.description.toLowerCase().includes(query)
        );
    }

    // Function to handle search button click or Enter key press
    function handleSearch() {
        const query = searchInput.value.trim();
        const filteredTools = filterTools(query);
        renderTools(filteredTools);
    }

    // Function to redirect to the specific calculator HTML page
    function redirectToCalculatorPage(toolId) {
        if (!toolId) return; // Exit if toolId is missing

        let filename;

        // --- Define Specific Filenames (if they don't match the ID pattern) ---
        // Add exceptions here if a tool's HTML file isn't simply "[id].html"
        if (toolId === 'compound-interest') {
            filename = 'compoundcalc.html';
        } else if (toolId === 'income-tax') {
            filename = 'tax-calc.html';
        } else if (toolId === 'sip') {
             filename = 'sip-calculator.html'; // If you kept this specific name
        } else if (toolId === 'health-insurance'){
            filename = 'health.html'
        } else if (toolId === 'pension') {
            filename = 'pc.html'
        } else if (toolId === 'savings') {
            filename = 'savings.html'
        } else if (toolId === 'hlv') {
            filename = 'hlc.html'
        } else if (toolId === 'ulip') {
            filename = 'ulip.html'
        }
        // Add more 'else if' blocks for other specific filenames as needed
        // else if (toolId === 'some-other-id') {
        //     filename = 'specific_page_name.html';
        // }

        // --- Default Filename Pattern ---
        else {
            // Assumes the HTML file is named exactly like the tool's id + .html
            // e.g., if id is 'savings', it looks for 'savings.html'
            // e.g., if id is 'term-insurance', it looks for 'term-insurance.html'
            filename = `${toolId}.html`;
        }

        console.log(`Redirecting to: ${filename}`); // For debugging
        window.location.href = filename; // Perform the redirection
    }

    // --- Base Event Listeners ---

    // Search functionality
    if (searchButton) {
        searchButton.addEventListener('click', handleSearch);
    }
    if (searchInput) {
        searchInput.addEventListener('keyup', function(event) {
            // Trigger search on Enter key press in the search input
            if (event.key === 'Enter') {
                handleSearch();
            }
        });
         // Optional: Live search as user types (uncomment if desired)
        // searchInput.addEventListener('input', handleSearch);
    }

    // --- Initial Render ---
    renderTools(financialTools); // Display all tools when the page loads

});