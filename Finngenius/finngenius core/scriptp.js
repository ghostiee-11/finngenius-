// scriptp.js

document.addEventListener('DOMContentLoaded', () => {
  const chatWidget = document.getElementById('chat-widget');
  const chatInterface = document.getElementById('chat-interface');
  const closeChatButton = document.getElementById('close-chat');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-button');

  // --- Session ID Generation (Vanilla JS) ---
  let chatSessionId;
  try {
      // Use crypto.randomUUID() if available (modern browsers)
      chatSessionId = crypto.randomUUID();
  } catch (e) {
      // Fallback for older environments (less unique but functional)
      console.warn("crypto.randomUUID() not available, using fallback.");
      chatSessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }
  console.log("Generated Chat Session ID:", chatSessionId); // For debugging

  // --- Explicit Backend URL ---
  // IMPORTANT: This MUST match the address and port where your Flask backend is running
  const BACKEND_CHAT_URL = 'http://127.0.0.1:5012/chat';

  // --- Check for Essential UI Elements ---
  if (!chatWidget || !chatInterface || !closeChatButton || !chatMessages || !chatInput || !sendButton) {
      console.error("One or more chat UI elements are missing from the HTML!");
      // Optionally display an error to the user or disable the chat feature
      if(chatWidget) chatWidget.style.display = 'none'; // Hide widget if interface is broken
      return; // Stop script execution
  }

  // --- Chat Visibility ---
  chatWidget.addEventListener('click', () => {
      chatInterface.classList.remove('hidden');
      setTimeout(() => chatInput.focus(), 50); // Focus after transition starts
  });

  closeChatButton.addEventListener('click', () => {
      chatInterface.classList.add('hidden');
  });

  // --- Add Message to UI ---
  function addMessage(sender, text) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('message');
      messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
      messageDiv.textContent = text; // Safely set text content
      chatMessages.appendChild(messageDiv);
      chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
  }

  // --- Send Message Logic ---
  async function sendMessage() {
      const messageText = chatInput.value.trim();
      if (!messageText) return; // Don't send empty messages

      addMessage('user', messageText);
      const currentInput = messageText; // Store before clearing
      chatInput.value = '';
      chatInput.disabled = true;
      sendButton.disabled = true;

      // Add thinking indicator
      const thinkingDiv = document.createElement('div');
      thinkingDiv.classList.add('message', 'bot-message');
      thinkingDiv.textContent = 'FinnyBot is thinking... 🐼';
      thinkingDiv.id = 'thinking-indicator'; // ID to easily remove it
      chatMessages.appendChild(thinkingDiv);
      chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

      try {
          console.log(`Sending to ${BACKEND_CHAT_URL} with session ${chatSessionId}: ${currentInput}`); // Debug Fetch
          const response = await fetch(BACKEND_CHAT_URL, {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
                  // No 'Origin' header needed here - browser handles it.
                  // Server needs 'Access-Control-Allow-Origin' header in response.
              },
              body: JSON.stringify({
                  message: currentInput,
                  session_id: chatSessionId
               })
          });

          // Remove thinking indicator regardless of outcome
          const thinkingIndicator = document.getElementById('thinking-indicator');
          if (thinkingIndicator) thinkingIndicator.remove(); // Use remove() which is simpler

          if (!response.ok) {
              // Handle HTTP errors (like 404, 500, 400)
              let errorMsg = `Sorry, FinnyBot had a problem reaching the server (Error ${response.status}).`;
              try {
                  // Try to get more specific error from backend JSON response
                  const errorData = await response.json();
                  if(errorData && errorData.reply) errorMsg += ` Server says: ${errorData.reply}`;
                  else if (errorData && errorData.error) errorMsg += ` Server says: ${errorData.error}`;
              } catch (e) {
                  // If response wasn't JSON or couldn't be parsed
                  console.warn("Could not parse error response body:", e);
                  errorMsg += ` Response: ${await response.text().catch(() => '(Could not read response text)')}`; // Attempt to get raw text
              }
              console.error('Chat API Error:', response.status, response.statusText, errorMsg);
              addMessage('bot', errorMsg);
              return; // Stop processing on error
          }

          // Handle successful response (2xx status code)
          const data = await response.json();
          if (data && data.reply) {
              addMessage('bot', data.reply);
          } else {
               // Handle cases where response is ok but data is missing reply
               addMessage('bot', 'Sorry, I received an unexpected response format from the server.');
               console.warn("Unexpected backend response format:", data);
          }

      } catch (error) {
           // Handle network errors (fetch couldn't complete, DNS issues, CORS blocked etc.)
           const thinkingIndicator = document.getElementById('thinking-indicator');
           if (thinkingIndicator) thinkingIndicator.remove(); // Ensure indicator is removed

          console.error('Error sending message (Network/Fetch):', error);
          // Provide helpful feedback for common issues like CORS
          if (error instanceof TypeError && error.message.toLowerCase().includes('fetch')) {
               // This often indicates a CORS issue or network problem
               addMessage('bot', 'Oops! Could not connect to FinnyBot. Please check the server is running and CORS is configured correctly.');
               console.error("Hint: Ensure the backend at " + BACKEND_CHAT_URL + " has CORS configured to allow requests from your origin (which might be 'null' for local files).");
          } else {
               // Generic network error
               addMessage('bot', 'Oops! Could not connect to FinnyBot. Please check your internet connection and try again.');
          }
      } finally {
          // Always re-enable input fields after attempt completes
          chatInput.disabled = false;
          sendButton.disabled = false;
          chatInput.focus(); // Focus back on input for usability
      }
  }

  // --- Event Listeners ---
  sendButton.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter only
          event.preventDefault(); // Prevent default form submission/newline
          sendMessage();
      }
  });

  // --- Placeholder for Floating Coins JS ---
  const floatingCoinsContainer = document.getElementById('floatingCoins');
  if (floatingCoinsContainer) {
      // Your JavaScript logic to create and animate floating coins would go here.
      // Example (very basic concept):
      // function createCoin() {
      //     const coin = document.createElement('div');
      //     coin.classList.add('coin-animation-class'); // CSS class for styling/animation
      //     coin.style.left = Math.random() * 100 + 'vw';
      //     coin.style.animationDuration = (Math.random() * 5 + 3) + 's';
      //     floatingCoinsContainer.appendChild(coin);
      //     setTimeout(() => coin.remove(), 8000); // Remove after animation
      // }
      // setInterval(createCoin, 1000); // Create a coin every second

      console.log("Floating coins container ready for animation logic.");
  }

}); // End DOMContentLoaded