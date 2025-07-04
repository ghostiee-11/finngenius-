document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const DKT_API_BASE_URL = 'http://127.0.0.1:5009'; // DKT Server URL
    const CHATBOT_API_BASE_URL = 'http://127.0.0.1:5001'; // Original Chatbot URL
    const DEFAULT_LEARNING_GOAL = 'C7'; // e.g., 'How to Choose Equity Funds'

    // --- UI Elements ---
    // General Chatbot Elements
    const chatMessagesDiv = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const chatSendButton = document.getElementById('chatSendButton');
    const chatLoading = document.getElementById('chatLoading');
    // Case Studies
    const caseStudiesListDiv = document.getElementById('case-studies-list');
    // Adaptive Quiz Elements
    const startQuizButton = document.getElementById('startQuizButton'); // Correct ID
    const quizInterfaceDiv = document.getElementById('quizInterface');
    const quizMessagesDiv = document.getElementById('quizMessages');   // Target for quiz content
    const quizLoading = document.getElementById('quizLoading');       // Target for quiz loading

    // --- User and DKT State ---
    let generalUserId = localStorage.getItem('finedu_user_id'); // For general chat
    if (!generalUserId) {
        generalUserId = `web-${crypto.randomUUID()}`;
        localStorage.setItem('finedu_user_id', generalUserId);
    }
    console.log("General FinEdu User ID:", generalUserId);

    let dktUserId = localStorage.getItem('finedu_dkt_user_id'); // Specific ID for DKT session
    let currentDktState = null; // e.g., 'QUIZ', 'LEARNING', 'GOAL_REACHED'
    let currentDktItem = null; // Stores the data of the current question/module (renamed from currentItemData for clarity)

    // --- Utility Functions ---
    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
             .replace(/&/g, "&") // Must be first
             .replace(/</g, "<")
             .replace(/>/g, ">")
             .replace(/"/g, "")
             .replace(/'/g, "'");
    }

    // --- Message Display Functions ---

    // Generic function to add HTML content to a target div
    function addContentToDiv(targetDiv, senderClass, contentHtml) {
        if (!targetDiv) return; // Exit if target div doesn't exist
        const messageElement = document.createElement('div');
        // Add base class and sender-specific class
        messageElement.classList.add(targetDiv === quizMessagesDiv ? 'quiz-message' : 'chat-message', senderClass);
        messageElement.innerHTML = contentHtml; // Assumes contentHtml is safe or already escaped
        targetDiv.appendChild(messageElement);
        // Scroll the specific container
        targetDiv.scrollTo({ top: targetDiv.scrollHeight, behavior: 'smooth' });
    }

    // Function to process text (escape, markdown, etc.) and return HTML string
    function processTextMessage(text) {
        let processedMessage = escapeHtml(text); // Escape first
        // Basic Markdown/Formatting
        processedMessage = processedMessage.replace(/^[\s]*[\*\-][\s]+(.*)/gm, '<li>$1</li>');
        processedMessage = processedMessage.replace(/(?:^|\n)(<li>.*?<\/li>)/gs, '$1');
        processedMessage = processedMessage.replace(/(<li>.*?<\/li>\s*)+/g, (match) => `<ul>${match.replace(/<\/li>\s*<li>/g,'</li><li>')}</ul>`);
        processedMessage = processedMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        processedMessage = processedMessage.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Convert Newlines to <br> (outside lists)
        const parts = processedMessage.split(/(<\/?ul>|<\/?li>)/g);
        const processedParts = parts.map((part, index) => {
            if (index % 2 === 0) return part.replace(/\n/g, '<br>');
            return part;
        });
        processedMessage = processedParts.join('');
        // Convert URLs
        const urlRegex = /(https?:\/\/[^\s<>"']+)/g;
        processedMessage = processedMessage.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        return processedMessage;
    }

    // Add a standard text message to the *General Chat*
    function addChatMessage(sender, message) {
        const processedHtml = processTextMessage(message);
        addContentToDiv(chatMessagesDiv, sender, processedHtml);
    }

    // Add a standard text message to the *Quiz Area*
    function addQuizMessage(sender, message) {
        const processedHtml = processTextMessage(message);
        // Use 'system', 'user', 'bot' classes appropriately even in quiz area for potential styling
        addContentToDiv(quizMessagesDiv, sender, processedHtml);
    }

    // Add potentially complex HTML (like DKT questions/modules) to the *Quiz Area*
    function addQuizHtmlContent(htmlContent) {
         // This function bypasses standard processing, assumes htmlContent is ready
         // It doesn't add sender classes automatically, structure within htmlContent should handle it
         if (!quizMessagesDiv) return;
         const wrapper = document.createElement('div'); // Use a simple wrapper
         wrapper.innerHTML = htmlContent;
         quizMessagesDiv.appendChild(wrapper);
         quizMessagesDiv.scrollTo({ top: quizMessagesDiv.scrollHeight, behavior: 'smooth' });
    }

    function scrollToBottom(element) {
        if(element) {
            element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' });
        }
    }

    // --- Loading Indicators ---
    function setChatLoading(isLoading) {
        if (chatLoading) chatLoading.style.display = isLoading ? 'block' : 'none';
    }

    function setQuizLoading(isLoading) {
        if (quizLoading) quizLoading.style.display = isLoading ? 'block' : 'none';
    }

    // Disable general chat input (e.g., during DKT interaction)
    function disableChatInput(disabled) {
        if (chatInput) {
            chatInput.disabled = disabled;
            chatInput.placeholder = disabled ? "Please complete the quiz step first..." : "Ask a general finance question...";
        }
        if (chatSendButton) chatSendButton.disabled = disabled;
    }

    // --- Case Studies Function (Keep as is) ---
    function loadCaseStudies() {
        // ... (your existing loadCaseStudies function - no changes needed)
        const caseStudiesFilePath = 'case_studies.json';
        fetch(caseStudiesFilePath)
            .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) return response.json();
                throw new TypeError("Received non-JSON response for case studies");
            })
            .then(data => {
                if (!caseStudiesListDiv) return;
                caseStudiesListDiv.innerHTML = '';
                if (!data || !Array.isArray(data) || data.length === 0) {
                    caseStudiesListDiv.innerHTML = '<p>No case studies available.</p>';
                    return;
                }
                data.forEach(study => {
                    const item = document.createElement('div');
                    item.className = 'case-study-item';
                    const title = study.title || "Untitled";
                    const summary = study.summary || "No summary.";
                    const source = study.source || 'N/A';
                    const date = study.scraped_date || 'N/A';
                    const link = study.link || '#';

                    const titleEl = document.createElement('h3');
                    const linkEl = document.createElement('a');
                    linkEl.href = link;
                    linkEl.target = '_blank';
                    linkEl.rel = 'noopener noreferrer';
                    linkEl.textContent = escapeHtml(title);
                    titleEl.appendChild(linkEl);

                    const sourceEl = document.createElement('p');
                    sourceEl.className = 'source';
                    sourceEl.textContent = `Source: ${escapeHtml(source)} | Fetched: ${escapeHtml(date)}`;

                    const summaryEl = document.createElement('p');
                    summaryEl.className = 'summary';
                    summaryEl.textContent = escapeHtml(summary);

                    item.appendChild(titleEl);
                    item.appendChild(sourceEl);
                    item.appendChild(summaryEl);
                    caseStudiesListDiv.appendChild(item);
                });
            })
            .catch(error => {
                console.error('Error loading case studies:', error);
                if (caseStudiesListDiv) caseStudiesListDiv.innerHTML = `<p style="color: var(--error-red, #dc3545);">Error loading studies: ${escapeHtml(error.message)}.</p>`;
            });
    }


    // --- DKT Interaction Functions ---

    async function startDktSession(goalConceptId = DEFAULT_LEARNING_GOAL) {
        if (!quizMessagesDiv || !quizInterfaceDiv || !startQuizButton) return;

        addQuizMessage('system', `Starting adaptive quiz (Goal: ${goalConceptId})...`);
        setQuizLoading(true);
        disableChatInput(true); // Disable general chat
        quizInterfaceDiv.style.display = 'block'; // Show the quiz area
        startQuizButton.style.display = 'none'; // Hide the start button
        quizMessagesDiv.innerHTML = ''; // Clear previous quiz content

        try {
            const response = await fetch(`${DKT_API_BASE_URL}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: "Web User",
                    age: 25, // Could be dynamic if collected
                    learning_goal: goalConceptId,
                }),
            });

            setQuizLoading(false);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            if (data.user_id && data.next_item) {
                dktUserId = data.user_id;
                localStorage.setItem('finedu_dkt_user_id', dktUserId);
                console.log("DKT Session Started. User ID:", dktUserId);
                processDktNextItem(data.next_item);
            } else {
                throw new Error("Invalid response from /start endpoint.");
            }

        } catch (error) {
            console.error('Error starting DKT session:', error);
            addQuizMessage('system', `Error starting quiz: ${escapeHtml(error.message)}`);
            setQuizLoading(false);
            disableChatInput(false); // Re-enable general chat on error
            if(startQuizButton) startQuizButton.style.display = 'block'; // Show start button again
            if(quizInterfaceDiv) quizInterfaceDiv.style.display = 'none'; // Hide quiz area
        }
    }

    async function sendDktAnswer(itemId, itemType, selectedOptionKey) {
        if (!dktUserId || !currentDktItem) {
             addQuizMessage('system', "Error: Quiz session or current item data not found.");
             return;
        }

        // Find the specific question container within quizMessagesDiv
        const questionContainer = quizMessagesDiv.querySelector(`#dkt-item-${itemId}`);
        if (questionContainer) {
             questionContainer.querySelectorAll('button').forEach(btn => {
                 btn.disabled = true; // Disable buttons immediately
                 btn.style.opacity = '0.7'; // Visually indicate disabled state
             });
        }
         setQuizLoading(true); // Show loading indicator in quiz area

        try {
            const response = await fetch(`${DKT_API_BASE_URL}/answer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: dktUserId,
                    item_id: itemId,
                    item_type: itemType,
                    answer: selectedOptionKey,
                }),
            });

             setQuizLoading(false);
             const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            // Display feedback within the question's container
             if (data.feedback && questionContainer) {
                 displayDktFeedback(data.feedback, questionContainer);
             } else {
                 addQuizMessage('system', `Answer processed.`); // Simple confirmation in quiz area
             }

             // Give a slight delay before showing the next item? Optional.
             // await new Promise(resolve => setTimeout(resolve, 500));

            if (data.next_item) {
                processDktNextItem(data.next_item);
            } else {
                 addQuizMessage('system', "Finished step, but no next item received. Requesting next...");
                 await getNextDktItem();
            }

        } catch (error) {
            console.error('Error sending DKT answer:', error);
            addQuizMessage('system', `Error processing answer: ${escapeHtml(error.message)}`);
             setQuizLoading(false);
             disableChatInput(false); // Re-enable general chat on error
        }
    }

     async function getNextDktItem() {
         if (!dktUserId) {
             addQuizMessage('system', "Error: Quiz session not active.");
             return;
         }
         addQuizMessage('system', `Fetching next step...`);
         setQuizLoading(true);
         disableChatInput(true); // Keep general chat disabled

         try {
             const response = await fetch(`${DKT_API_BASE_URL}/get_next_item`, {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({ user_id: dktUserId }),
             });

             setQuizLoading(false);
             const data = await response.json();

             if (!response.ok) {
                 throw new Error(data.error || `Server error: ${response.status}`);
             }

             if (data.next_item) {
                 processDktNextItem(data.next_item);
             } else {
                 throw new Error("No next item received from /get_next_item.");
             }

         } catch (error) {
             console.error('Error getting next DKT item:', error);
             addQuizMessage('system', `Error fetching next step: ${escapeHtml(error.message)}`);
             setQuizLoading(false);
             disableChatInput(false); // Re-enable general chat on error
         }
     }

    // --- DKT UI Rendering Functions (Append to quizMessagesDiv) ---

    function displayDktQuestion(itemData) {
        currentDktItem = itemData;
        currentDktState = 'AWAITING_ANSWER';
        disableChatInput(true);

        // Build HTML string for the question container
        let questionHtml = `<div class="dkt-item-container" id="dkt-item-${itemData.item_id}">`;
        const questionType = itemData.type === 'quiz_question' ? 'Quiz' : 'Practice';
        questionHtml += `<p><strong>${questionType} Question:</strong><br>${escapeHtml(itemData.text)}</p>`;
        questionHtml += `<div class="dkt-options-list">`;

        if (itemData.options && typeof itemData.options === 'object') {
             Object.entries(itemData.options).forEach(([key, value]) => {
                 // Embed the onclick directly, calling sendDktAnswer with necessary params
                 questionHtml += `<button class="dkt-option-button" data-option-key="${key}" onclick="window.handleDktOptionClick('${itemData.item_id}', '${itemData.type}', '${key}')">`;
                 questionHtml += `${key}. ${escapeHtml(value)}`;
                 questionHtml += `</button>`;
             });
        } else {
             questionHtml += `<p>Error: Options not available.</p>`;
        }

        questionHtml += `</div></div>`; // Close options-list and container

        // Add the fully constructed HTML to the quiz area
        addQuizHtmlContent(questionHtml);
    }

    // Make handleDktOptionClick globally accessible for inline onclick handlers
    window.handleDktOptionClick = (itemId, itemType, key) => {
         // Optional: Add visual feedback directly on click if desired
         const button = quizMessagesDiv.querySelector(`#dkt-item-${itemId} button[data-option-key="${key}"]`);
         if (button) button.style.borderWidth = '2px'; // Example feedback
         sendDktAnswer(itemId, itemType, key);
    };

    function displayDktFeedback(feedbackData, questionContainer) {
         if (!questionContainer) return; // Safety check

         const resultPara = document.createElement('p');
         resultPara.className = 'dkt-result';

         if (feedbackData.correct) {
             resultPara.textContent = `Correct! 👍`;
             resultPara.style.color = 'var(--success-green, #22C55E)';
         } else {
             resultPara.textContent = `Not quite. The correct answer was ${feedbackData.correct_option}. Keep learning! 🧠`;
             resultPara.style.color = 'orange';
         }

         // Style the buttons (already disabled by sendDktAnswer)
         questionContainer.querySelectorAll('.dkt-option-button').forEach(btn => {
             const btnKey = btn.getAttribute('data-option-key');
             if (btnKey === feedbackData.correct_option) {
                 btn.classList.add('correct');
             } else if (btnKey === feedbackData.selected_option) {
                 btn.classList.add('incorrect');
             } else {
                  btn.style.opacity = '0.6'; // Fade out non-selected, non-correct options
             }
         });

         questionContainer.appendChild(resultPara);
         scrollToBottom(quizMessagesDiv); // Scroll quiz area specifically
    }

    function displayDktModule(itemData) {
        currentDktItem = itemData;
        currentDktState = 'READING_MODULE';
        disableChatInput(true);

        let moduleHtml = `<div class="dkt-item-container module-content" id="dkt-item-${itemData.item_id}">`;
        moduleHtml += `<h3>Module: ${escapeHtml(itemData.name)}</h3>`;

        if (itemData.concepts && itemData.concepts.length > 0) {
            moduleHtml += `<p><strong>Concepts Covered:</strong> ${escapeHtml(itemData.concepts.join(', '))}</p>`;
        }

        if (itemData.content_sections && itemData.content_sections.length > 0) {
            moduleHtml += `<div class="module-sections">`;
            itemData.content_sections.forEach(section => {
                 // Process each section like a message for basic formatting
                 moduleHtml += `<p>${processTextMessage(section)}</p>`;
            });
            moduleHtml += `</div>`;
        }

        // Add the continue button with an inline onclick handler
        moduleHtml += `<button class="dkt-continue-button" onclick="window.handleDktContinueClick(this)">Continue Learning</button>`;
        moduleHtml += `</div>`; // Close container

        addQuizHtmlContent(moduleHtml);
    }

    // Make handleDktContinueClick globally accessible
    window.handleDktContinueClick = (buttonElement) => {
         buttonElement.textContent = 'Loading next step...';
         buttonElement.disabled = true;
         getNextDktItem();
    };


     function displayDktEndMessage(itemData) {
         currentDktItem = itemData;
         currentDktState = itemData.type;
         disableChatInput(false); // Re-enable general chat

         let endHtml = `<div class="dkt-item-container dkt-end-message">`;
         endHtml += `<p><strong>${escapeHtml(itemData.message)}</strong></p>`;

         if (itemData.knowledge_state) {
             endHtml += '<p><em>Final Knowledge State Estimate:</em></p>';
             endHtml += '<ul class="knowledge-state-list">';
             Object.entries(itemData.knowledge_state)
                .sort(([, probA], [, probB]) => probB - probA)
                .forEach(([conceptName, probability]) => {
                    let style = '';
                    if (probability >= 0.85) style = 'color: var(--success-green, #198754);';
                    else if (probability < 0.5) style = 'color: var(--text-muted, #6c757d);';
                    endHtml += `<li style="${style}">${escapeHtml(conceptName)}: ${Math.round(probability * 100)}%</li>`;
             });
             endHtml += '</ul>';
         }

          // Add restart button
         endHtml += `<button class="action-button" style="margin-top: 15px;" onclick="window.handleDktRestartClick()">Start New Quiz</button>`;

         endHtml += `</div>`; // Close container

         addQuizHtmlContent(endHtml);

         // Show the main start button again
         if(startQuizButton) startQuizButton.style.display = 'block';
     }

     // Make restart handler global
     window.handleDktRestartClick = () => {
          dktUserId = null; localStorage.removeItem('finedu_dkt_user_id');
          currentDktState = null; currentDktItem = null;
          addQuizMessage('system', "Starting a new adaptive quiz...");
          if(quizInterfaceDiv) quizInterfaceDiv.style.display = 'none'; // Hide interface briefly
          startDktSession(); // Restart with default goal
     };


    // --- Central DKT Item Processor ---
    function processDktNextItem(nextItemData) {
        if (!nextItemData || !nextItemData.type) {
            addQuizMessage('system', "Error: Received invalid next item data from backend.");
            disableChatInput(false); // Re-enable general chat if stuck
            if(startQuizButton) startQuizButton.style.display = 'block'; // Show start button again
            return;
        }

        console.log("Processing DKT Next Item:", nextItemData);

        switch (nextItemData.type) {
            case 'quiz_question':
            case 'practice_question':
                displayDktQuestion(nextItemData);
                break;
            case 'module_content':
                displayDktModule(nextItemData);
                break;
            case 'goal_reached':
            case 'finished_roadmap':
                displayDktEndMessage(nextItemData);
                break;
            case 'error':
                addQuizMessage('system', `Error from backend: ${escapeHtml(nextItemData.message)}`);
                disableChatInput(false);
                if(startQuizButton) startQuizButton.style.display = 'block'; // Show start button again
                break;
            default:
                addQuizMessage('system', `Received unknown item type: ${escapeHtml(nextItemData.type)}`);
                disableChatInput(false);
                if(startQuizButton) startQuizButton.style.display = 'block'; // Show start button again
        }
    }

    // --- General Chatbot Function (Remains largely the same) ---
    async function sendMessageToGeneralBot() {
        if (!chatInput || !chatMessagesDiv) return; // Check elements exist
        const userInput = chatInput.value.trim();
        if (!userInput) return;

        // Prevent general chat if a DKT action is pending UI interaction
        if (currentDktState === 'AWAITING_ANSWER' || currentDktState === 'READING_MODULE') {
            // Add message to the *general* chat area, not the quiz area
            addChatMessage('system', "Please complete the current quiz/learning step first.");
            return;
        }

        addChatMessage('user', userInput); // Adds to #chatMessages
        chatInput.value = '';
        setChatLoading(true); // Use chat loading indicator
        disableChatInput(true); // Disable general chat input

        try {
            const response = await fetch(`${CHATBOT_API_BASE_URL}/api/chat`, { // Ensure this endpoint is correct
                method: 'POST',
                headers: { 'Content-Type': 'application/json', },
                body: JSON.stringify({ user_input: userInput, user_id: generalUserId, current_context: "General FinEdu Chat" }),
            });

            setChatLoading(false);
             // Re-enable general chat ONLY if DKT is not active
             if (!currentDktState || currentDktState === 'GOAL_REACHED' || currentDktState === 'FINISHED_ROADMAP') {
                 disableChatInput(false);
             }


            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.answer) { addChatMessage('bot', data.answer); } // Add bot response to general chat
            else { addChatMessage('bot', "Sorry, I didn't get a valid response."); }
            // Handle any specific quiz/links from the general bot if needed here

        } catch (error) {
            console.error('Error sending/receiving general chat message:', error);
            addChatMessage('bot', `Sorry, an error occurred: ${escapeHtml(error.message)}`); // Error to general chat
             setChatLoading(false);
             // Re-enable general chat ONLY if DKT is not active
             if (!currentDktState || currentDktState === 'GOAL_REACHED' || currentDktState === 'FINISHED_ROADMAP') {
                 disableChatInput(false);
             }
        } finally {
             if(chatInput) chatInput.focus();
        }
    }

    // --- Event Listeners ---
    if (startQuizButton) {
        startQuizButton.addEventListener('click', () => {
             startDktSession(); // Start with the default goal
        });
    }

    // General Chat Input Listeners
    if (chatSendButton) chatSendButton.addEventListener('click', sendMessageToGeneralBot);
    if (chatInput) chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessageToGeneralBot();
        }
    });

    // --- Initial Load ---
    loadCaseStudies(); // Load case studies on page load

    // Add initial welcome message to *general* chat
    addChatMessage('system', "Welcome to FinEdu! Ask a general financial question below, browse case studies, or click 'Start Adaptive Quiz' to begin a personalized lesson.");

    disableChatInput(false); // Ensure general chat input is enabled initially
    setChatLoading(false);
    setQuizLoading(false);

    // Optional: Check if a DKT session was active and try to resume?
    // More complex - would need backend state checking. For now, we just check if ID exists.
    // if (dktUserId) {
    //     addQuizMessage('system', "Detected a previous quiz session ID. Click 'Start Adaptive Quiz' to potentially resume or start over.");
    //     // Note: Resuming requires backend logic to handle existing user ID in /start
    //     // or a dedicated /resume endpoint.
    // }

});