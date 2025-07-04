document.addEventListener('DOMContentLoaded', () => {
    loadAllArticles();
    // Initialize Lucide icons if they are used on this page
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        lucide.createIcons();
    }
});

async function loadAllArticles() {
    const container = document.getElementById('all-articles-list');
    // Use the same API endpoint URL as index.html (make sure port is correct)
    const apiUrl = 'http://localhost:5001/api/finance-news'; // Adjust port if needed

    if (!container) {
        console.error('Error: Container #all-articles-list not found.');
        return;
    }

    // Show loading state (already in HTML, but clear previous errors)
    container.innerHTML = '<p class="loading-message">Loading articles...</p>';

    try {
        console.log(`Fetching all articles from: ${apiUrl}`);
        const response = await fetch(apiUrl);
        console.log(`API Response Status: ${response.status}`);

        if (!response.ok) {
            let errorData = { message: `HTTP error! Status: ${response.status}` };
            try {
                const jsonError = await response.json();
                if (jsonError && jsonError.error) {
                    errorData.message = jsonError.error;
                    if (jsonError.details) errorData.details = jsonError.details;
                }
            } catch (e) {}
            throw new Error(`Failed to fetch articles: ${errorData.message}${errorData.details ? ` (${errorData.details})` : ''}`);
        }

        const articles = await response.json();

        // Clear loading message/skeletons
        container.innerHTML = '';

        if (!Array.isArray(articles)) {
            throw new Error("Invalid data format: Expected an array of articles from API.");
        }

        if (articles.length === 0) {
            container.innerHTML = '<p class="error-message">No articles available at the moment.</p>';
            return;
        }

        // Render all articles
        articles.forEach(article => {
            if (!article || typeof article.title !== 'string' || typeof article.url !== 'string' || !article.url) {
                console.warn("Skipping invalid article data:", article);
                return; // Skip this article
            }

            const articleElement = document.createElement('div');
            articleElement.className = 'all-article-item'; // Add a class for styling

            const linkElement = document.createElement('a');
            linkElement.href = article.url;
            linkElement.target = '_blank';
            linkElement.rel = 'noopener noreferrer';
            linkElement.textContent = article.title; // Display title as the link text

            articleElement.appendChild(linkElement);
            container.appendChild(articleElement);
        });

    } catch (error) {
        console.error('Error loading or processing all articles:', error);
        container.innerHTML = `<p class="error-message">Could not load articles. ${error.message}</p>`;
    }
}