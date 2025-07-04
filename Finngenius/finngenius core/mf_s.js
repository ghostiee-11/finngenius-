document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const API_BASE_URL = 'http://localhost:5005/api';
    const EXPLAIN_URL = `${API_BASE_URL}/explain`;
    const ILLUSTRATE_URL = `${API_BASE_URL}/illustrate`;
    const QUIZ_QUESTIONS_URL = `${API_BASE_URL}/quiz/questions`;

    // --- Get Elements ---
    // Explainer Section
    const termInput = document.getElementById('termInput');
    const explainButton = document.getElementById('explainButton');
    const loadingExplain = document.getElementById('loadingExplain');
    const explanationArea = document.getElementById('explanationArea');

    // Illustrator Section
    const profileButtons = document.querySelectorAll('.profile-button');
    const loadingIllustrate = document.getElementById('loadingIllustrate');
    const illustratorOutputArea = document.getElementById('illustratorOutput');
    const illustratorMessage = document.getElementById('illustratorMessage');
    const illustratorResultsArea = document.getElementById('illustratorResultsArea');

    // Quiz Section
    const quizQuestionsDiv = document.getElementById('quizQuestions');
    const loadingQuiz = document.getElementById('loadingQuiz'); // Loading indicator inside questions div
    const quizFeedbackArea = document.getElementById('quizFeedbackArea');
    const quizActionButton = document.getElementById('quizActionButton'); // Renamed button
    const quizResultArea = document.getElementById('quizResultArea');

    // --- State Variables ---
    let loadedQuizQuestions = [];
    let currentQuestionIndex = 0;
    let score = 0;
    let quizActive = false; // To control button actions

    // --- Utility Functions (setLoadingState, displayMessage, clearAndHide, apiRequest) ---
    // ... (Keep these functions as they were in the previous version) ...
    function setLoadingState(loadingElement, isLoading, elementsToDisable = []) {
        if (loadingElement) {
            loadingElement.style.display = isLoading ? 'block' : 'none';
        }
        elementsToDisable.forEach(el => {
            if (el) el.disabled = isLoading;
        });
    }
    function displayMessage(message, targetArea, isError = false) {
        if (!targetArea) return;
        targetArea.innerHTML = '';
        const messageP = document.createElement('p');
        messageP.textContent = message;
        if (isError) {
            messageP.classList.add('error-text');
            messageP.textContent = `Error: ${message}`;
        } else {
             messageP.classList.remove('error-text');
        }
        targetArea.appendChild(messageP);
        targetArea.style.display = 'block';
    }
    function clearAndHide(targetArea) {
        if (!targetArea) return;
        targetArea.innerHTML = '';
        targetArea.style.display = 'none';
    }
    async function apiRequest(url, method = 'GET', body = null) {
        const options = {
            method,
            headers: {},
        };
        if (body) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(body);
        }
        try {
            const response = await fetch(url, options);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data?.error || response.statusText || `Server error: ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error(`API Request Error (${method} ${url}):`, error);
            if (error instanceof SyntaxError) {
                 throw new Error(`Invalid response from backend at ${url}. Expected JSON.`);
            } else if (error instanceof TypeError) {
                 throw new Error(`Network Error: Could not reach backend at ${url}. Ensure it's running.`);
            }
            throw error;
        }
    }


    // --- Explain Term Functionality ---
    // ... (Keep handleExplainRequest function as is) ...
    async function handleExplainRequest() {
        const term = termInput.value.trim();
        if (!term) {
            displayMessage('Please enter a term or question.', explanationArea, true);
            return;
        }
        clearAndHide(explanationArea);
        setLoadingState(loadingExplain, true, [termInput, explainButton]);
        try {
            const data = await apiRequest(EXPLAIN_URL, 'POST', { term });
            displayMessage(data.explanation, explanationArea, false);
        } catch (error) {
            displayMessage(error.message, explanationArea, true);
        } finally {
            setLoadingState(loadingExplain, false, [termInput, explainButton]);
        }
    }

    // --- Illustrate Examples Functionality ---
    // ... (Keep handleIllustrateRequest and displayIllustratorResults functions as is) ...
        async function handleIllustrateRequest(event) {
        const selectedButton = event.target.closest('.profile-button');
        if (!selectedButton) return;

        const profile = selectedButton.dataset.profile;
        if (!profile) {
            displayMessage('Could not determine selected profile.', illustratorMessage, true);
            return;
        }

        profileButtons.forEach(btn => btn.classList.remove('active'));
        selectedButton.classList.add('active');

        clearAndHide(illustratorResultsArea);
        clearAndHide(illustratorMessage);
        setLoadingState(loadingIllustrate, true, Array.from(profileButtons));

        try {
            const data = await apiRequest(ILLUSTRATE_URL, 'POST', { profile });
            displayMessage(data.message || 'Received examples.', illustratorMessage, false);
            displayIllustratorResults(data.examples);
        } catch (error) {
            displayMessage(error.message, illustratorMessage, true);
        } finally {
            setLoadingState(loadingIllustrate, false, Array.from(profileButtons));
        }
    }
    function displayIllustratorResults(examples) {
        clearAndHide(illustratorResultsArea);
        if (!examples || !Array.isArray(examples) || examples.length === 0) {
             displayMessage('No examples found for this profile.', illustratorMessage, false);
             illustratorResultsArea.style.display = 'none';
            return;
        }
        const table = document.createElement('table');
        table.id = 'resultsTable';
        table.classList.add('results-table');
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        const headers = Object.keys(examples[0]);
        headers.forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        const tbody = table.createTBody();
        examples.forEach(example => {
            const row = tbody.insertRow();
            headers.forEach(key => {
                const cell = row.insertCell();
                cell.textContent = example[key] !== null && example[key] !== undefined ? example[key] : 'N/A';
            });
        });
        illustratorResultsArea.appendChild(table);
        illustratorResultsArea.style.display = 'block';
    }

    // --- Quiz Functionality ---

    /**
     * Loads quiz questions and starts the quiz.
     */
    async function loadQuiz() {
        setLoadingState(loadingQuiz, true, [quizActionButton]); // Show loading inside question area
        clearAndHide(quizResultArea);
        clearAndHide(quizFeedbackArea);
        quizQuestionsDiv.innerHTML = ''; // Clear previous content before showing loader
        quizQuestionsDiv.appendChild(loadingQuiz); // Ensure loader is visible
        loadedQuizQuestions = [];
        currentQuestionIndex = 0;
        score = 0;
        quizActive = false; // Quiz not active until questions are loaded

        try {
            const questionsData = await apiRequest(QUIZ_QUESTIONS_URL, 'GET');

            if (!Array.isArray(questionsData) || questionsData.length === 0) {
                displayMessage('No quiz questions available.', quizQuestionsDiv, false);
                return; // Stop if no questions
            }

            loadedQuizQuestions = questionsData;
            quizActive = true; // Quiz is now ready
            displayCurrentQuestion(); // Display the first question

        } catch (err) {
            displayMessage(`Failed to load quiz: ${err.message}`, quizQuestionsDiv, true);
        } finally {
            // Hide loading indicator (it was appended to quizQuestionsDiv)
            setLoadingState(loadingQuiz, false);
            // Button state is handled by displayCurrentQuestion or error state
             if (!quizActive) {
                 quizActionButton.disabled = true; // Ensure button disabled if load failed
             }
        }
    }

    /**
     * Displays the question at the currentQuestionIndex.
     */
    function displayCurrentQuestion() {
        if (!quizActive || currentQuestionIndex >= loadedQuizQuestions.length) {
            displayFinalScore(); // Should not happen if logic is correct, but handles end case
            return;
        }

        const q = loadedQuizQuestions[currentQuestionIndex];
        quizQuestionsDiv.innerHTML = ''; // Clear previous question/loader
        clearAndHide(quizFeedbackArea); // Clear previous feedback

        const qDiv = document.createElement('div');
        qDiv.classList.add('quiz-question-single'); // Add a class for styling single question

        const questionLabel = document.createElement('p');
        questionLabel.innerHTML = `<strong>Q${currentQuestionIndex + 1} / ${loadedQuizQuestions.length}:</strong> `;
        questionLabel.appendChild(document.createTextNode(q.question || ''));
        qDiv.appendChild(questionLabel);

        const optionsDiv = document.createElement('div');
        optionsDiv.classList.add('quiz-options');

        if (Array.isArray(q.options)) {
            q.options.forEach((opt) => {
                const safeOptValue = String(opt).replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '');
                const optionId = `q${currentQuestionIndex}-option-${safeOptValue}`;
                const label = document.createElement('label');
                label.setAttribute('for', optionId);

                const input = document.createElement('input');
                input.type = 'radio';
                input.name = `question${currentQuestionIndex}`; // Unique name per question ensures only one selection
                input.value = opt;
                input.id = optionId;

                label.appendChild(input);
                label.appendChild(document.createTextNode(` ${opt}`));
                optionsDiv.appendChild(label);
                optionsDiv.appendChild(document.createElement('br'));
            });
        } else {
            optionsDiv.textContent = 'No options available.';
        }
        qDiv.appendChild(optionsDiv);
        quizQuestionsDiv.appendChild(qDiv);

        // Reset button for the new question
        quizActionButton.textContent = 'Submit Answer';
        quizActionButton.disabled = false;
        quizActionButton.dataset.action = 'submit'; // Set action state
    }

    /**
     * Handles the click event for the main quiz action button.
     */
    function handleQuizAction() {
        if (!quizActive) return; // Do nothing if quiz isn't loaded/active

        const action = quizActionButton.dataset.action;

        if (action === 'submit') {
            handleSubmitAnswer();
        } else if (action === 'next') {
            handleNextQuestion();
        }
    }

    /**
     * Processes the submitted answer for the current question.
     */
    function handleSubmitAnswer() {
        const selectedOption = quizQuestionsDiv.querySelector(`input[name="question${currentQuestionIndex}"]:checked`);

        if (!selectedOption) {
            displayMessage('Please select an answer.', quizFeedbackArea, true); // Show temporary error in feedback area
            return;
        }

        clearAndHide(quizFeedbackArea); // Clear selection warning if present
        const userAnswer = selectedOption.value;
        const correctAnswer = loadedQuizQuestions[currentQuestionIndex].answer;

        // Disable radio buttons for the current question
        const radioButtons = quizQuestionsDiv.querySelectorAll(`input[name="question${currentQuestionIndex}"]`);
        radioButtons.forEach(rb => rb.disabled = true);

        // Display feedback
        if (userAnswer === correctAnswer) {
            score++;
            quizFeedbackArea.textContent = 'Correct!';
            quizFeedbackArea.className = 'feedback-correct'; // Add class for styling
        } else {
            quizFeedbackArea.textContent = `Incorrect. The correct answer was: ${correctAnswer}`;
            quizFeedbackArea.className = 'feedback-incorrect'; // Add class for styling
        }
        quizFeedbackArea.style.display = 'block';

        // Change button to "Next Question"
        quizActionButton.textContent = 'Next Question';
        quizActionButton.dataset.action = 'next'; // Change action state
        quizActionButton.disabled = false; // Ensure it's enabled
    }

    /**
     * Moves to the next question or displays the final score.
     */
    function handleNextQuestion() {
        currentQuestionIndex++;
        if (currentQuestionIndex < loadedQuizQuestions.length) {
            displayCurrentQuestion(); // Display next question
        } else {
            displayFinalScore(); // End of quiz
        }
    }

    /**
     * Displays the final quiz score.
     */
    function displayFinalScore() {
        quizActive = false; // Quiz finished
        quizQuestionsDiv.innerHTML = '<p>Quiz Complete!</p>'; // Clear last question
        clearAndHide(quizFeedbackArea);
        quizActionButton.style.display = 'none'; // Hide the action button

        let resultMessage = `Your final score is ${score} out of ${loadedQuizQuestions.length}.`;
        let suggestion = '';
        const percentage = (score / loadedQuizQuestions.length) * 100;

        if (percentage >= 80) {
            suggestion = "Excellent understanding!";
        } else if (percentage >= 50) {
            suggestion = "Good job! You have a solid grasp.";
        } else {
            suggestion = "Keep learning! Explore the Jargon Buster to improve.";
        }

        quizResultArea.innerHTML = `
            <div class="quiz-result-card">
                <h3>Quiz Results</h3>
                <p>${resultMessage}</p>
                <p>${suggestion}</p>
                <button id="retakeQuizButton">Retake Quiz</button> <!-- Optional: Add retake button -->
            </div>
        `;
        quizResultArea.style.display = 'block';

        // Add event listener for the optional retake button
        const retakeButton = document.getElementById('retakeQuizButton');
        if(retakeButton) {
            retakeButton.addEventListener('click', startNewQuiz);
        }
    }

    /**
     * Resets the quiz state and loads questions again.
     */
    function startNewQuiz() {
        quizActionButton.style.display = 'block'; // Show action button again
        loadQuiz(); // Reload everything
    }


    // --- Event Listeners Setup ---
    if (explainButton) {
        explainButton.addEventListener('click', handleExplainRequest);
    }
    if (termInput) {
        termInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleExplainRequest();
            }
        });
    }

    profileButtons.forEach(button => {
        button.addEventListener('click', handleIllustrateRequest);
    });

    // Connect the main action button
    if (quizActionButton) {
        quizActionButton.addEventListener('click', handleQuizAction);
    }

    // --- Initialization ---
    loadQuiz(); // Load quiz questions when the page loads
    if(termInput) termInput.focus();

}); // End DOMContentLoaded