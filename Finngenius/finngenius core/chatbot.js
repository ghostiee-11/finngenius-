// --- Configuration ---
const BACKEND_URL = 'http://localhost:5003'; // Base URL for backend

// --- DOM Elements ---
// IDs match the latest HTML structure
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chatInput');
const chatSendButton = document.getElementById('chatSendButton');
const chatClearButton = document.getElementById('chatClearButton');
const chatMessagesArea = document.getElementById('chatMessages');
const statusArea = document.getElementById('status-area');
const chatLoading = document.getElementById('chatLoading'); // Get loading indicator

// --- Event Listeners ---
chatForm.addEventListener('submit', handleQuerySubmit);
chatClearButton.addEventListener('click', handleClearChat);

// --- Functions ---

// Function to add a message to the chat display
function addMessageToChat(text, sender, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', sender === 'user' ? 'user' : 'bot');

    if (isError) {
        messageDiv.classList.add('error');
        // Error prefix is handled by CSS ::before
        messageDiv.textContent = text;
    } else {
        // Basic sanitization - consider a library like DOMPurify for robust protection
        const sanitizedText = text.replace(/</g, "<").replace(/>/g, ">");
        // Make links clickable (simple version)
        const linkifiedText = sanitizedText.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        messageDiv.innerHTML = linkifiedText.replace(/\n/g, '<br>'); // Render newlines
    }

    // Remove the initial "Ask a question..." message if it exists
    const initialMessage = chatMessagesArea.querySelector('em');
    if (initialMessage && initialMessage.parentNode.classList.contains('chat-message')) {
        initialMessage.parentNode.remove();
    }

    chatMessagesArea.appendChild(messageDiv);
    // Scroll to the bottom
    chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
}

// Handle query submission
async function handleQuerySubmit(event) {
    event.preventDefault();
    const userQuery = chatInput.value.trim();

    if (!userQuery) {
        showStatus('Please enter a question.', true);
        return;
    }

    addMessageToChat(userQuery, 'user');

    // Update UI: Start Loading
    showStatus(''); // Clear previous status
    chatLoading.style.display = 'block'; // Show loading indicator
    chatSendButton.disabled = true;
    chatClearButton.disabled = true;
    chatInput.value = '';
    chatInput.disabled = true;

    try {
        const response = await fetch(`${BACKEND_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', },
            body: JSON.stringify({ query: userQuery }),
        });

        // Clear loading state regardless of response ok status
        chatLoading.style.display = 'none';

        if (!response.ok) {
            let errorMsg = `Request failed: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) { errorMsg = errorData.error; }
            } catch (e) { /* Ignore if body isn't JSON */ }
            console.error('Backend request failed:', errorMsg);
            addMessageToChat(errorMsg, 'assistant', true);
        } else {
            const data = await response.json();
            if (data.answer) {
                addMessageToChat(data.answer, 'assistant');
            } else if (data.error) {
                 console.error('Backend returned an error:', data.error);
                 addMessageToChat(data.error, 'assistant', true);
            } else {
                 console.error('Unexpected response format:', data);
                 addMessageToChat('Received an unexpected response format from the server.', 'assistant', true);
            }
        }

    } catch (error) {
        console.error('Fetch error:', error);
        chatLoading.style.display = 'none'; // Hide loading on error
        addMessageToChat(`Network error: Could not connect to the backend (${error.message}). Please check if the backend is running.`, 'assistant', true);
    } finally {
        // Update UI: Finish Loading
        chatSendButton.disabled = false;
        chatClearButton.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
}

// Handle clearing the chat
async function handleClearChat() {
    showStatus('Clearing chat history...');
    chatLoading.style.display = 'block'; // Show loading indicator
    chatSendButton.disabled = true;
    chatClearButton.disabled = true;

    try {
        // Assuming backend memory needs clearing
        const response = await fetch(`${BACKEND_URL}/clear`, { method: 'POST' });

        chatLoading.style.display = 'none'; // Hide loading

        if (!response.ok) {
            let errorMsg = `Failed to clear chat on backend: ${response.status} ${response.statusText}`;
             try { const errorData = await response.json(); if (errorData && errorData.message) errorMsg += ` - ${errorData.message}`; } catch (e) {/*ignore*/}
            showStatus(errorMsg, true); // Show error in status
            console.error('Clear chat failed:', errorMsg);
        } else {
            chatMessagesArea.innerHTML = '<div class="chat-message bot"><em>Chat history cleared. Ask a new question!</em></div>'; // Clear frontend display
            clearStatus(); // Clear status message
            console.log("Chat history cleared successfully.");
        }
    } catch (error) {
        console.error('Clear chat fetch error:', error);
        chatLoading.style.display = 'none'; // Hide loading
        showStatus(`Network error clearing chat: ${error.message}`, true);
    } finally {
         chatSendButton.disabled = false;
         chatClearButton.disabled = false;
         chatInput.focus();
         // Ensure status is cleared if it wasn't an error shown above
         if (!statusArea.classList.contains('error')) {
             clearStatus();
         }
    }
}

// Function to show status messages
function showStatus(message, isError = false) {
    statusArea.textContent = message;
    statusArea.className = isError ? 'status error' : 'status';
}

// Function to clear the status area
function clearStatus() {
    statusArea.textContent = '';
    statusArea.className = 'status';
}