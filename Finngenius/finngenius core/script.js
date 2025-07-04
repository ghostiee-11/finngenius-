'use strict';

// Declare lucide and TradingView as global variables. This assumes that the libraries are loaded externally.
var lucide = lucide || {};
var TradingView = TradingView || {};
// Google Sign-In - will be initialized later
var google = google || {};

// Backend API Base URLs (Adjust if needed)
const AUTH_API_BASE_URL = 'http://localhost:5015/api/auth'; // For Login/Signup/Google Auth (Port 5015 as requested)
const DATA_API_BASE_URL = 'http://localhost:5001';      // For Ticker/News data (Using 5001 from original code)

document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide icons (Initial call)
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        lucide.createIcons();
    } else {
        console.warn("Lucide library not loaded or createIcons function unavailable.");
    }

    // Set current year in footer
    const currentYearElement = document.getElementById("current-year");
    if (currentYearElement) {
        currentYearElement.textContent = new Date().getFullYear();
    }

    // Navbar scroll effect
    const navbar = document.querySelector(".navbar");
    if (navbar) {
        window.addEventListener("scroll", () => {
            if (window.scrollY > 10) {
                navbar.classList.add("scrolled");
            } else {
                navbar.classList.remove("scrolled");
            }
        });
        // Initial check in case page loads scrolled
        if (window.scrollY > 10) navbar.classList.add("scrolled");
    }

    // Mobile menu toggle
    const mobileMenuButton = document.querySelector(".mobile-menu-button");
    const mobileMenuCloseButton = document.querySelector(".mobile-menu-close");
    const mobileMenu = document.querySelector(".mobile-menu");

    if (mobileMenuButton && mobileMenuCloseButton && mobileMenu) {
        mobileMenuButton.addEventListener("click", () => {
            mobileMenu.classList.add("active");
            document.body.style.overflow = "hidden"; // Prevent background scroll
        });

        mobileMenuCloseButton.addEventListener("click", () => {
            mobileMenu.classList.remove("active");
            document.body.style.overflow = ""; // Restore background scroll
        });

        // Close mobile menu on NAV link click (but not for auth trigger)
        mobileMenu.querySelectorAll('.mobile-nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                // Don't close if it's the auth trigger button inside the mobile menu
                if (!link.classList.contains('auth-trigger-btn') && !link.classList.contains('auth-action-btn')) { // Also check for logout btn
                    mobileMenu.classList.remove('active');
                    document.body.style.overflow = ""; // Restore scroll
                } else if (link.classList.contains('auth-trigger-btn')) {
                    // For the auth trigger, prevent default link behavior if it's an <a>
                    // The main listener for auth-trigger-btn will handle opening modal & closing menu
                    e.preventDefault();
                }
                // Allow logout click to proceed (handled by its own listener)
            });
        });
    }


    // Dropdown menu
    const dropdownTrigger = document.querySelector(".dropdown-trigger");
    const dropdown = document.querySelector(".dropdown");

    if (dropdownTrigger && dropdown) {
        dropdownTrigger.addEventListener("click", (e) => {
            e.stopPropagation(); // Prevent click from closing immediately
            dropdown.classList.toggle("active");
        });

        // Close dropdown when clicking outside
        document.addEventListener("click", (e) => {
            // If the click is outside the dropdown itself
            if (dropdown && !dropdown.contains(e.target)) {
                dropdown.classList.remove("active");
            }
        });
    }

    // Navbar hover effect
    const navLinks = document.querySelectorAll(".desktop-nav .nav-link, .desktop-nav .dropdown-trigger");
    const hoverIndicator = document.querySelector(".hover-indicator");
    const navContainer = document.querySelector(".desktop-nav");

    if (navLinks.length > 0 && hoverIndicator && navContainer) {
        navLinks.forEach((link) => {
            link.addEventListener("mouseenter", function () {
                // Don't show hover for non-interactive logged-in user display
                if (this.id === 'logged-in-user') return;

                const rect = this.getBoundingClientRect();
                const navRect = navContainer.getBoundingClientRect();

                hoverIndicator.style.width = `${rect.width}px`;
                hoverIndicator.style.left = `${rect.left - navRect.left}px`;
                hoverIndicator.style.opacity = "1";
            });
        });

        navContainer.addEventListener("mouseleave", () => {
            // Check if the mouse is *really* outside the nav container, not just moving between links
            setTimeout(() => {
                if (navContainer && !navContainer.matches(':hover')) {
                     hoverIndicator.style.opacity = "0";
                }
            }, 50); // Small delay to allow moving between items
        });
        // Ensure indicator hides if mouse leaves a link but stays over nav bg briefly
        navLinks.forEach(link => {
             link.addEventListener("mouseleave", (e) => {
                 // Check relatedTarget to see if mouse moved to another nav item or outside
                 if (navContainer && !navContainer.contains(e.relatedTarget)) {
                     hoverIndicator.style.opacity = "0";
                 }
             });
        });
    }


    // --- START: Authentication Modal Logic (Integrated) ---
    const authModal = document.getElementById('auth-modal');
    const authTriggerBtns = document.querySelectorAll('.auth-trigger-btn'); // These trigger the modal

    const closeModalBtn = authModal?.querySelector('.close-button');
    const loginView = document.getElementById('login-view');
    const signupView = document.getElementById('signup-view');
    const showSignupLink = document.getElementById('show-signup');
    const showLoginLink = document.getElementById('show-login');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const googleSignupBtn = document.getElementById('google-signup-btn');
    const githubSignupBtn = document.getElementById('github-signup-btn'); // GitHub still placeholder

    // --- Authentication Helper Functions ---

    function showAuthError(formElement, message) {
        if (!formElement) return;
        clearAuthError(formElement); // Clear previous error first
        let errorElement = document.createElement('p');
        errorElement.className = 'error-message';
        errorElement.style.color = '#ff4d4d'; // Red color for error
        errorElement.style.textAlign = 'center';
        errorElement.style.marginTop = '10px';
        errorElement.style.fontSize = '0.9em';
        errorElement.textContent = message;

        const submitButton = formElement.querySelector('button[type="submit"]');
        const socialDivider = formElement.querySelector('.social-login-divider'); // Reference point

        if (submitButton) {
             // Insert between primary button and OR divider/social buttons
             submitButton.parentNode.insertBefore(errorElement, socialDivider || submitButton.nextSibling);
        } else {
             formElement.appendChild(errorElement); // Fallback
        }
    }

    function clearAuthError(formElement) {
         if (!formElement) return;
         const errorElement = formElement.querySelector('.error-message');
         if (errorElement) {
             errorElement.remove();
         }
         // Clear input error styling if it was added
        formElement.querySelectorAll('input.error').forEach(input => {
            input.classList.remove('error');
            input.style.borderColor = ''; // Reset specific style if applied
        });
        // Also clear potential red borders from password mismatch
        formElement.querySelectorAll('input[type="password"]').forEach(input => {
             input.style.borderColor = ''; // Reset border style
        });
    }


    // Function to handle successful login/signup
    function handleAuthSuccess(token, userData) {
        console.log('Auth successful:', userData);
        if (!token || !userData || !userData.email) {
            console.error("Incomplete data received on auth success.");
            alert("Login failed: Invalid response from server.");
            handleLogout(); // Ensure clean state
            return;
        }
        localStorage.setItem('authToken', token); // Store the JWT
        localStorage.setItem('userData', JSON.stringify(userData)); // Store user info
        updateUIAfterAuthStateChange(); // Update navbar, etc.
        closeModal(); // Close the login/signup modal
    }

    // Function to handle logout
    function handleLogout(event) {
        // Prevent default if it's attached to an <a> tag
        if (event) event.preventDefault();

        localStorage.removeItem('authToken');
        localStorage.removeItem('userData');
        updateUIAfterAuthStateChange(); // Update UI back to logged-out state
        console.log('User logged out');
        // Optional: Maybe show a brief "Logged out" message
        // Optional: redirect to home or refresh
        // window.location.href = '/';
    }

    // Function to update UI based on login state
    function updateUIAfterAuthStateChange() {
        const token = localStorage.getItem('authToken');
        const userDataString = localStorage.getItem('userData');
        let userData = null;
         try {
            // Only parse if data exists
            userData = userDataString ? JSON.parse(userDataString) : null;
         } catch(e) {
             console.error("Error parsing user data from localStorage:", e);
             // Clear potentially corrupted data if parsing fails
             localStorage.removeItem('authToken');
             localStorage.removeItem('userData');
             // Ensure token is also considered null if data is corrupt
             token = null;
         }

        const loggedOutElements = document.querySelectorAll('.logged-out-item');
        const loggedInElements = document.querySelectorAll('.logged-in-item');
        const desktopUserDisplay = document.getElementById('logged-in-user');
        const mobileUserDisplay = document.getElementById('mobile-logged-in-user');
        const logoutButtons = document.querySelectorAll('.auth-action-btn'); // Select all logout buttons/links


        if (token && userData && userData.email) { // Check for token and valid user data
            // --- Logged In State ---
            loggedOutElements.forEach(el => el.style.display = 'none');
            // Use 'flex' for mobile nav items which are likely flex containers
            loggedInElements.forEach(el => {
                 el.style.display = el.closest('.mobile-nav') ? 'flex' : 'inline-block';
            });


            const welcomeText = `Hi, ${userData.name || userData.email.split('@')[0]}`;
            if (desktopUserDisplay) desktopUserDisplay.textContent = welcomeText;

            if (mobileUserDisplay) {
                 // Ensure the icon stays if it exists
                const userIcon = mobileUserDisplay.querySelector('i[data-lucide="user"]');
                mobileUserDisplay.textContent = welcomeText + ' '; // Add space before icon if needed
                 if (userIcon) mobileUserDisplay.appendChild(userIcon); // Append icon after text
            }

            // Ensure logout handler is attached to all logout buttons/links
            logoutButtons.forEach(button => {
                 if (!button.dataset.listenerAttached) {
                    button.addEventListener('click', handleLogout);
                    button.dataset.listenerAttached = 'true';
                }
            });

            console.log("UI updated to: Logged In");

        } else {
            // --- Logged Out State ---
            // Use 'inline-flex' for desktop button, 'flex' for mobile link
            loggedOutElements.forEach(el => {
                el.style.display = el.closest('.mobile-nav') ? 'flex' : 'inline-flex';
            });
            loggedInElements.forEach(el => el.style.display = 'none');

            // Detach listener to prevent memory leaks if element might be re-added later
            // Or just rely on the check inside the 'if (token...)' block
             logoutButtons.forEach(button => {
                 if (button.dataset.listenerAttached) {
                    button.removeEventListener('click', handleLogout);
                    delete button.dataset.listenerAttached;
                }
            });


            console.log("UI updated to: Logged Out");
        }
    }


    // --- Google Sign-In Initialization ---
    function initializeGoogleSignIn() {
        if (typeof google === 'undefined' || !google.accounts || !google.accounts.id) {
            console.warn("Google Identity Services script not loaded yet or ID module unavailable.");
            return;
        }
        try {
            google.accounts.id.initialize({
                client_id: "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com", // <--- *** REPLACE THIS WITH YOUR ACTUAL CLIENT ID ***
                callback: handleGoogleCredentialResponse, // Function defined below to handle the ID token
                context: 'signup',
                // ux_mode: 'popup', // Let's rely on the callback triggered by the button for popup
            });
            console.log("Google Sign In initialized.");

        } catch (error) {
            console.error("Error initializing Google Sign In:", error);
             if (googleSignupBtn) {
                googleSignupBtn.disabled = true;
                googleSignupBtn.title = "Google Sign-In unavailable";
             }
             if(authModal?.classList.contains('active')) {
                const activeForm = signupView?.style.display !== 'none' ? signupForm : loginForm;
                 if(activeForm) showAuthError(activeForm, "Google Sign-In unavailable.");
             }
        }
    }
    // Delay initialization slightly to increase chances of GIS script being ready
    window.setTimeout(initializeGoogleSignIn, 500); // Adjust delay if needed


    // --- Google Sign-In Callback (Handles ID Token) ---
    async function handleGoogleCredentialResponse(response) {
        console.log("Received Google ID Token Credential Response");
        const idToken = response.credential;
        if (!idToken) {
            console.error("Google response missing credential token.");
            const activeForm = signupView?.style.display !== 'none' ? signupForm : loginForm;
            if(activeForm && authModal?.classList.contains('active')) showAuthError(activeForm, 'Google Sign-In failed (missing token).');
            else alert("Google Sign-In failed (missing token).");
            return;
        }

        // Send this idToken to your backend for verification
        try {
            const res = await fetch(`${AUTH_API_BASE_URL}/google`, { // Use AUTH URL
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: idToken }),
            });

            const data = await res.json();

            if (!res.ok || !data.success) { // Check backend success flag
                console.error("Backend error during Google Sign-In:", data);
                const errorMsg = data.message || `Google Sign-In Error (${res.status})`;
                const activeForm = signupView?.style.display !== 'none' ? signupForm : loginForm;
                if(activeForm && authModal?.classList.contains('active')) showAuthError(activeForm, errorMsg);
                else alert(`Google Sign-In Failed: ${errorMsg}`);
                return;
            }

            // Backend returns JWT and user data on success
            handleAuthSuccess(data.token, data.user);

        } catch (error) {
            console.error('Network or fetch error sending Google token to backend:', error);
            const activeForm = signupView?.style.display !== 'none' ? signupForm : loginForm;
             if(activeForm && authModal?.classList.contains('active')) showAuthError(activeForm, 'Network error during Google Sign-In.');
             else alert('Google Sign-In Failed: Network error.');
        }
    }

    // --- Event Listener for Custom Google Button Click ---
    googleSignupBtn?.addEventListener('click', () => {
         if (!signupForm) return;
         clearAuthError(signupForm);
         console.log("Custom Google signup button clicked");

         if (typeof google !== 'undefined' && google.accounts && google.accounts.id) {
            // Directly trigger the One Tap / Sign in with Google prompt
            // The `handleGoogleCredentialResponse` callback (defined above)
            // will handle the ID token when the user signs in via the prompt.
            google.accounts.id.prompt((notification) => {
                if (notification.isNotDisplayed()) {
                    console.warn(`Google prompt not displayed: ${notification.getNotDisplayedReason()}`);
                    // Maybe show a message to the user if the prompt is blocked
                     showAuthError(signupForm, "Google Sign-In prompt might be blocked. Please check browser settings.");
                } else if (notification.isSkippedMoment()) {
                     console.warn(`Google prompt skipped: ${notification.getSkippedReason()}`);
                } else if (notification.isDismissedMoment()) {
                     console.warn(`Google prompt dismissed: ${notification.getDismissedReason()}`);
                }
            });
         } else {
            console.error("Google Sign-In (google.accounts.id) is not ready or initialized.");
            showAuthError(signupForm, "Google Sign-In is not ready. Please wait or refresh.");
        }
    });


    // --- Modal Opening/Closing & View Switching ---
    const openModal = () => {
        if (authModal) {
            clearAuthError(loginForm);
            clearAuthError(signupForm);
            authModal.classList.add('active');
            document.body.style.overflow = "hidden";
            if (typeof lucide !== 'undefined' && lucide.createIcons) {
                 // Re-render icons specifically within the modal each time it opens
                 // This ensures icons added dynamically (like in error messages) render.
                lucide.createIcons({ nodes: authModal.querySelectorAll('[data-lucide]') });
            }
            showLoginView(); // Default view
        }
    };
    const closeModal = () => {
        if (authModal) {
            authModal.classList.remove('active');
            document.body.style.overflow = "";
        }
    };
    const showLoginView = () => {
        if (loginView && signupView) {
            loginView.style.display = 'block';
            signupView.style.display = 'none';
            clearAuthError(signupForm); // Clear errors when switching away
        }
    };
    const showSignupView = () => {
         if (loginView && signupView) {
            loginView.style.display = 'none';
            signupView.style.display = 'block';
             clearAuthError(loginForm); // Clear errors when switching away
        }
    };
    // Attach listeners to ALL auth trigger buttons
    authTriggerBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            openModal();
             // If the trigger button is inside the mobile menu, close the menu itself
            if (mobileMenu && mobileMenu.classList.contains('active') && btn.closest('.mobile-menu')) {
                 mobileMenu.classList.remove('active');
                 document.body.style.overflow = "hidden"; // Keep body locked
            }
        });
    });
    closeModalBtn?.addEventListener('click', closeModal);
    authModal?.addEventListener('click', (event) => { if (event.target === authModal) { closeModal(); } });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && authModal?.classList.contains('active')) { closeModal(); } });
    showSignupLink?.addEventListener('click', (e) => { e.preventDefault(); showSignupView(); });
    showLoginLink?.addEventListener('click', (e) => { e.preventDefault(); showLoginView(); });


    // --- Email/Password Form Submissions (Using Fetch to Backend) ---

    // LOGIN FORM
    loginForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!loginForm) return;
        clearAuthError(loginForm);

        const emailInput = document.getElementById('login-email');
        const passwordInput = document.getElementById('login-password');
        if (!emailInput || !passwordInput) { console.error("Login form inputs not found"); return; }

        const email = emailInput.value.trim();
        const password = passwordInput.value;
        if (!email || !password) { showAuthError(loginForm, 'Please enter both email and password.'); return; }

        try {
            const response = await fetch(`${AUTH_API_BASE_URL}/login`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (!response.ok || !data.success) { showAuthError(loginForm, data.message || `Login failed (${response.status})`); return; }
            handleAuthSuccess(data.token, data.user);
        } catch (error) {
            console.error('Login Fetch Error:', error);
            showAuthError(loginForm, 'Login failed. Network error or server unavailable.');
        }
    });

    // SIGNUP FORM
    signupForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        if(!signupForm) return;
        clearAuthError(signupForm);

        const nameInput = document.getElementById('signup-name');
        const emailInput = document.getElementById('signup-email');
        const passwordInput = document.getElementById('signup-password');
        const confirmPasswordInput = document.getElementById('signup-confirm-password');
        if (!nameInput || !emailInput || !passwordInput || !confirmPasswordInput) { console.error("Signup form inputs not found"); return; }

        const name = nameInput.value.trim();
        const email = emailInput.value.trim();
        const password = passwordInput.value;
        const confirmPassword = confirmPasswordInput.value;

        // Frontend validation
        if (!name || !email || !password || !confirmPassword) { showAuthError(signupForm, 'Please fill in all fields.'); return; }
        if (!/\S+@\S+\.\S+/.test(email)) { showAuthError(signupForm, 'Please enter a valid email address.'); return; }
        if (password.length < 6) {
             showAuthError(signupForm, 'Password must be at least 6 characters long.');
             passwordInput.style.borderColor = '#ff4d4d'; passwordInput.classList.add('error'); return;
        }
        if (password !== confirmPassword) {
            showAuthError(signupForm, 'Passwords do not match!');
            passwordInput.style.borderColor = '#ff4d4d'; passwordInput.classList.add('error');
            confirmPasswordInput.style.borderColor = '#ff4d4d'; confirmPasswordInput.classList.add('error');
            confirmPasswordInput.focus(); return;
        }

        try {
             const response = await fetch(`${AUTH_API_BASE_URL}/signup`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, email, password }),
            });
            const data = await response.json();
            if (!response.ok || !data.success) { showAuthError(signupForm, data.message || `Sign up failed (${response.status})`); return; }
             handleAuthSuccess(data.token, data.user); // Auto-login after signup
        } catch (error) {
            console.error('Signup Fetch Error:', error);
            showAuthError(signupForm, 'Sign up failed. Network error or server unavailable.');
        }
    });


    // --- GitHub Signup Placeholder ---
    githubSignupBtn?.addEventListener('click', () => {
        console.log('Sign up with GitHub clicked (Placeholder)');
        alert('GitHub Sign up not implemented yet.');
        // TODO: Implement GitHub OAuth flow (requires backend setup too)
    });
    // --- END: Authentication Modal Logic ---


    // --- Initialize TradingView Widget ---
    const tvWidgetContainer = document.getElementById("tradingview-widget");
    if (tvWidgetContainer && typeof TradingView !== 'undefined' && TradingView.widget) {
        try {
            new TradingView.widget({ /* ... TradingView config ... */
                autosize: true, symbol: "BSE:SENSEX", interval: "D", timezone: "Asia/Kolkata",
                theme: "dark", style: "1", locale: "in", toolbar_bg: "rgba(30, 30, 30, 0)",
                enable_publishing: false, hide_top_toolbar: true, hide_legend: true, save_image: false,
                container_id: "tradingview-widget", backgroundColor: "rgba(18, 18, 18, 0)",
                gridColor: "rgba(39, 39, 42, 0.5)",
             });
        } catch (error) {
            console.error("Error initializing TradingView Widget:", error);
            tvWidgetContainer.innerHTML = "<p style='color: var(--muted-foreground, #888); text-align: center; padding: 2rem;'>Could not load market chart.</p>";
        }
    } else if (tvWidgetContainer) {
        console.warn("TradingView library not loaded or widget function unavailable.");
        tvWidgetContainer.innerHTML = "<p style='color: var(--muted-foreground, #888); text-align: center; padding: 2rem;'>Market chart unavailable.</p>";
    }


    // --- START: Stock Ticker Logic (Using DATA_API_BASE_URL) ---
    const tickerSymbols = [
        { id: 'ticker-nifty', symbol: '^NSEI', name: 'NIFTY 50' }, { id: 'ticker-sensex', symbol: '^BSESN', name: 'SENSEX' },
        { id: 'ticker-banknifty', symbol: '^NSEBANK', name: 'BANKNIFTY' }, { id: 'ticker-dow', symbol: '^DJI', name: 'DOW JONES' },
        { id: 'ticker-btc', symbol: 'BTC-USD', name: 'BTC' }, { id: 'ticker-eth', symbol: 'ETH-USD', name: 'ETH' }
    ];
    const tickerUpdateInterval = 15000;
    let lastPrices = {};

    async function fetchTickerPrice(symbol) {
        // Ensure DATA_API_BASE_URL is defined
        if (!DATA_API_BASE_URL) {
             console.error("DATA_API_BASE_URL is not defined.");
             return null;
        }
        const url = `${DATA_API_BASE_URL}/api/stock-price/${encodeURIComponent(symbol)}`;
        try {
            const response = await fetch(url);
            if (!response.ok) {
                if (response.status !== 404 && response.status !== 503) { console.warn(`Ticker fetch failed for ${symbol}: ${response.status}`); }
                return null;
            }
            const data = await response.json();
            if (data && typeof data.price === 'number' && typeof data.currency === 'string') { return data; }
            else { console.warn(`Invalid price/currency data received for ${symbol}:`, data); return null; }
        } catch (error) {
            if (!(error instanceof TypeError && error.message.includes('Failed to fetch'))) { console.error(`Network or other error fetching ticker price for ${symbol}:`, error); }
            return null;
        }
    }

    function updateTickerUI(item, data) {
        const allTickerElements = document.querySelectorAll(`.ticker-item`);
        allTickerElements.forEach(tickerElement => {
            const nameElement = tickerElement.querySelector('.name');
            if (!nameElement || nameElement.textContent !== item.name) { return; }

            const priceElement = tickerElement.querySelector('.price');
            const changeElement = tickerElement.querySelector('.change');
            if (!priceElement || !changeElement) return;

            priceElement.classList.remove('loading');
            changeElement.classList.remove('loading', 'positive', 'negative', 'neutral');

            if (data && data.price !== undefined && data.currency !== undefined) {
                let formattedPrice; let currencySymbol = '';
                switch (data.currency.toUpperCase()) {
                    case 'USD': currencySymbol = '$'; formattedPrice = data.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); break;
                    case 'INR': currencySymbol = '₹'; formattedPrice = data.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); break;
                    default: currencySymbol = data.currency + ' '; formattedPrice = data.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                }
                priceElement.textContent = currencySymbol + formattedPrice;

                const lastPrice = lastPrices[item.symbol];
                if (lastPrice !== undefined && data.price !== lastPrice) { changeElement.classList.add(data.price > lastPrice ? 'positive' : 'negative'); }
                else { changeElement.classList.add('neutral'); }
                lastPrices[item.symbol] = data.price;
            } else { priceElement.textContent = 'N/A'; changeElement.textContent = ''; }
        });
    }

    async function fetchAllTickerData() {
        const promises = tickerSymbols.map(item => fetchTickerPrice(item.symbol));
        const results = await Promise.allSettled(promises);
        results.forEach((result, index) => {
            const item = tickerSymbols[index];
            if (result.status === 'fulfilled' && result.value) { updateTickerUI(item, result.value); }
            else { updateTickerUI(item, null); }
        });
    }

    const tickerContainer = document.querySelector('.stock-ticker-container');
    if (tickerContainer) {
        fetchAllTickerData(); // Initial fetch
        setInterval(fetchAllTickerData, tickerUpdateInterval); // Periodic fetch
    }
    // --- END: Stock Ticker Logic ---


    // Load articles (Using DATA_API_BASE_URL)
    loadArticles();

    // --- Initial UI Check on Load ---
    updateUIAfterAuthStateChange(); // Check login state and update UI immediately

}); // End DOMContentLoaded Listener


// ==================================================
// == Function to load articles from API == (Uses DATA_API_BASE_URL)
// ==================================================
async function loadArticles() {
    const articlesContainer = document.getElementById("articles-container");
    // Ensure DATA_API_BASE_URL is defined
    if (!DATA_API_BASE_URL) {
         console.error("DATA_API_BASE_URL is not defined for loading articles.");
         if (articlesContainer) articlesContainer.innerHTML = `<p style="color: var(--muted-foreground, #888); grid-column: 1 / -1; text-align: center;">Could not load articles. Configuration error.</p>`;
         return;
    }
    const apiUrl = `${DATA_API_BASE_URL}/api/finance-news`;

    if (!articlesContainer) { console.error("Article container (#articles-container) not found."); return; }

    // Keep skeletons until data is loaded or error occurs
    // articlesContainer.innerHTML = ""; // Don't clear skeletons yet

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            let errorText = `HTTP error! Status: ${response.status}`;
            try { const errorJson = await response.json(); if (errorJson && errorJson.error) { errorText = `API Error: ${errorJson.error}${errorJson.details ? ` (${errorJson.details})` : ''}`; } }
            catch (e) { /* Ignore if response body is not JSON */ }
             throw new Error(errorText);
        }
        const articles = await response.json();

        articlesContainer.innerHTML = ""; // Clear skeletons NOW

        if (!Array.isArray(articles)) { throw new Error("Invalid data format: Expected an array of articles."); }
        if (articles.length === 0) { articlesContainer.innerHTML = '<p style="color: var(--muted-foreground, #888); grid-column: 1 / -1; text-align: center;">No recent articles found.</p>'; return; }

        const maxArticlesToShow = 6;
        const articlesToDisplay = articles.slice(0, maxArticlesToShow);

        articlesToDisplay.forEach((article, index) => {
            if (!article || typeof article.title !== 'string' || !article.title.trim() || typeof article.url !== 'string' || !article.url.startsWith('http')) {
                 console.warn("Skipping invalid article data:", article); return;
            }
            const articleCard = document.createElement("a");
            articleCard.href = article.url; articleCard.className = "article-card animate-fade-up";
            articleCard.style.animationDelay = `${index * 100}ms`; articleCard.target = "_blank"; articleCard.rel = "noopener noreferrer";
            articleCard.innerHTML = `
                <div class="article-content">
                    <h3 class="article-title">${article.title}</h3>
                    ${article.source ? `<span class="article-source">${article.source}</span>` : ''}
                    ${article.publishedAt ? `<span class="article-date">${new Date(article.publishedAt).toLocaleDateString()}</span>` : ''}
                </div>
                <div class="article-footer">
                    <span class="article-link">Read more <i data-lucide="arrow-right"></i></span>
                </div>`;
            articlesContainer.appendChild(articleCard);
        });

        if (typeof lucide !== 'undefined' && lucide.createIcons) {
             // Render only newly added icons for performance
             lucide.createIcons({ nodes: articlesContainer.querySelectorAll('[data-lucide]') });
        }
    } catch (error) {
        console.error('Error loading or processing articles:', error);
        // Ensure skeletons are cleared before showing error
        if (articlesContainer) articlesContainer.innerHTML = `<p style="color: var(--muted-foreground, #888); grid-column: 1 / -1; text-align: center;">Could not load articles. ${error.message || 'Please try again later.'}</p>`;
    }
}