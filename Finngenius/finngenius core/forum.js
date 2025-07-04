document.addEventListener('DOMContentLoaded', () => {
    // --- Constants and Globals ---
    const LOCAL_STORAGE_THREADS_KEY = 'localChanThreads';
    const LOCAL_STORAGE_USER_KEY = 'localChanUser';
    const MAX_LOCALSTORAGE_APPROX_MB = 4.5; // Be conservative with images
    // IMPORTANT: Add actual bad words here for filtering to work effectively
    const BAD_WORDS = ['examplebadword', 'testprofanity', 'shibboleth', /* Add more words, lowercase */ ];

    let currentThreads = [];
    let currentUser = 'Anonymous';
    let mediaDataUrlCache = { newThread: null, reply: null };

    // --- DOM Element Getters (Using functions for robustness) ---
    const getElem = (id) => document.getElementById(id);
    const pageContentDiv = getElem('page-content');
    const currentUserDisplay = getElem('current-user-display');
    const changeUserBtn = getElem('change-user-btn');
    const userModal = getElem('user-modal');
    const closeUserModalBtn = getElem('close-user-modal-btn');
    const newUsernameInput = getElem('new-username');
    const setUserBtn = getElem('set-user-btn');
    const cancelUserBtn = getElem('cancel-user-btn');
    const landingPageContentDiv = getElem('landing-page-content');
    const newThreadForm = getElem('new-thread-form');
    const threadAuthorInput = getElem('thread-author');
    const threadSubjectInput = getElem('thread-subject');
    const threadContentInput = getElem('thread-content');
    const threadMediaInput = getElem('thread-media');
    const threadMediaPreview = getElem('thread-media-preview');
    const submitThreadBtn = getElem('submit-thread-btn');
    const threadsListDiv = getElem('threads-list');
    const formStatusDiv = getElem('form-status');
    const singleThreadPageContentDiv = getElem('single-thread-page-content');
    const singleThreadContainer = getElem('single-thread-container');
    const singleThreadSubjectH2 = getElem('single-thread-subject');
    const replyFormContainer = getElem('reply-form-container');
    const replyForm = getElem('reply-form');
    const replyParentIdInput = getElem('reply-parent-id');
    const replyAuthorInput = getElem('reply-author');
    const replyContentInput = getElem('reply-content');
    const replyMediaInput = getElem('reply-media');
    const replyMediaPreview = getElem('reply-media-preview');
    const submitReplyBtn = getElem('submit-reply-btn');
    const replyFormStatusDiv = getElem('reply-form-status');

    // --- Initialization ---
    function initializeForum() {
        if (!pageContentDiv || !landingPageContentDiv || !singleThreadPageContentDiv || !threadsListDiv) {
             console.error("CRITICAL ERROR: Core page structure elements missing. Check HTML IDs.");
             alert("Forum initialization failed. Check console (F12).");
             return;
        }
        console.log("Initializing Forum...");
        loadCurrentUser();
        updateUserDisplay();
        loadThreads();
        setupEventListeners();
        handleRouting();
        checkStorageUsage();
        console.log("Forum Initialized.");
    }

    // --- Routing ---
    function handleRouting() {
        const hash = window.location.hash;
        console.log("Routing based on hash:", hash);
        if (!landingPageContentDiv || !singleThreadPageContentDiv) {
             console.error("Routing error: Page content divs not found."); return;
        }

        if (hash.startsWith('#thread=')) {
            const threadId = hash.substring(8);
            landingPageContentDiv.classList.add('hidden');
            singleThreadPageContentDiv.classList.remove('hidden');
            renderSingleThreadPage(threadId);
            window.scrollTo(0, 0); // Scroll to top when viewing a thread
        } else {
            landingPageContentDiv.classList.remove('hidden');
            singleThreadPageContentDiv.classList.add('hidden');
            renderLandingPage();
        }
    }
    window.addEventListener('hashchange', handleRouting);

    // --- View Rendering ---
    function renderLandingPage() {
        console.log("Rendering Landing Page");
        if (!threadsListDiv) return;
        threadsListDiv.innerHTML = '';
        if (currentThreads.length === 0) {
            threadsListDiv.innerHTML = '<p class="no-threads-message">No threads yet. Start one!</p>';
        } else {
             const sortedThreads = [...currentThreads].sort((a, b) => {
                const lastReplyTimeA = a.replies.length ? a.replies[a.replies.length - 1].timestamp : 0;
                const lastReplyTimeB = b.replies.length ? b.replies[b.replies.length - 1].timestamp : 0;
                const lastActivityA = Math.max(a.opPost?.timestamp || 0, lastReplyTimeA); // Add safety checks
                const lastActivityB = Math.max(b.opPost?.timestamp || 0, lastReplyTimeB); // Add safety checks
                return lastActivityB - lastActivityA;
             });
             sortedThreads.forEach(thread => {
                if (thread && thread.opPost) {
                     const opElement = createPostElement(thread.opPost, true, false);
                     if (opElement) threadsListDiv.appendChild(opElement); // Append only if element created
                } else { console.warn("Skipping invalid thread object:", thread); }
            });
        }
        if (newThreadForm) newThreadForm.reset();
        if (threadMediaPreview) clearMediaPreview(threadMediaPreview);
        mediaDataUrlCache.newThread = null;
        if (formStatusDiv) showStatus("", "info", formStatusDiv);
    }

    function renderSingleThreadPage(threadId) {
        console.log(`Rendering Single Thread Page for ID: ${threadId}`);
        if (!singleThreadContainer || !singleThreadSubjectH2 || !replyFormContainer || !replyParentIdInput) {
             console.error("Single thread page elements missing!"); return;
        }
        const thread = currentThreads.find(t => t.id === threadId);
        singleThreadContainer.innerHTML = '';

        if (thread && thread.opPost) {
            singleThreadSubjectH2.textContent = thread.opPost.subject || `Thread ${thread.id.substring(1)}`; // Show shorter ID
            const threadElement = createThreadElement(thread, true);
            if (threadElement) singleThreadContainer.appendChild(threadElement);

            replyParentIdInput.value = threadId;
            if (replyForm) replyForm.reset();
            if (replyMediaPreview) clearMediaPreview(replyMediaPreview);
            mediaDataUrlCache.reply = null;
            if (replyFormStatusDiv) showStatus("", "info", replyFormStatusDiv);
            replyFormContainer.classList.remove('hidden');
        } else {
            console.error(`Thread ${threadId} not found!`);
            singleThreadSubjectH2.textContent = "Thread Not Found";
            singleThreadContainer.innerHTML = '<p class="error-message" style="color: var(--error-red); text-align:center; padding: 20px;">Could not find the requested thread.</p>';
            replyFormContainer.classList.add('hidden');
        }
    }

    // --- Data Handling & Sample Data ---
    function loadThreads() { /* ... (same as previous step, includes getDefaultThreads logic) ... */
        console.log("Loading threads...");
        let storedThreads = localStorage.getItem(LOCAL_STORAGE_THREADS_KEY);
        try {
            let parsedData = storedThreads ? JSON.parse(storedThreads) : [];
            if (!Array.isArray(parsedData) || parsedData.length === 0) {
                 console.log("No valid threads in localStorage or empty array, loading defaults.");
                 currentThreads = getDefaultThreads();
                 saveThreads();
             } else {
                 currentThreads = parsedData;
            }
            currentThreads.sort((a, b) => (b.opPost?.timestamp || 0) - (a.opPost?.timestamp || 0));
            console.log(`Loaded ${currentThreads.length} threads.`);
        } catch (error) {
            console.error("Error parsing threads from localStorage:", error);
            currentThreads = getDefaultThreads();
            localStorage.removeItem(LOCAL_STORAGE_THREADS_KEY);
            saveThreads();
            showStatus("Error loading threads. Storage cleared. Defaults loaded.", "error", formStatusDiv);
        }
     }
    function getDefaultThreads() { /* ... (same as previous step) ... */
        console.log("Generating default threads...");
        const now = Date.now();
        const placeholderImageDataUrl = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"; // 1x1 transparent gif

        return [
            {
                 id: `T${now-300000}`, opPost: { id: `P${now-300000}`, parentId: null, author: 'Admin', subject: 'Welcome & Rules', content: 'Welcome to LocalChan!\n\n1. Be excellent to each other (locally).\n2. Keep images small.\n3. Data *only* saved in your browser.', mediaDataUrl: null, timestamp: now - 300000 }, replies: [
                    { id: `P${now-250000}`, parentId: `T${now-300000}`, threadId: `T${now-300000}`, author: 'FirstUser', subject: null, content: 'Got it! Seems cool.', mediaDataUrl: null, timestamp: now - 250000 },
                ]
            },
             {
                 id: `T${now-200000}`, opPost: { id: `P${now-200000}`, parentId: null, author: 'Picasso', subject: 'Trying an Image', content: 'Does this tiny image work?', mediaDataUrl: placeholderImageDataUrl, timestamp: now - 200000 }, replies: [
                     { id: `P${now-180000}`, parentId: `T${now-200000}`, threadId: `T${now-200000}`, author: 'Anonymous', subject: null, content: '>>P'+(now-200000)+'\nYep, see the placeholder!', mediaDataUrl: null, timestamp: now - 180000 }, // Example of quoting OP
                 ]
             },
             {
                 id: `T${now-100000}`, opPost: { id: `P${now-100000}`, parentId: null, author: 'Tester', subject: 'Filtering Test', content: 'How about this: examplebadword shibboleth', mediaDataUrl: null, timestamp: now - 100000 }, replies: []
             },
             {
                 id: `T${now-50000}`, opPost: { id: `P${now-50000}`, parentId: null, author: 'Anon', subject: 'Long Post Test', content: 'This is just a slightly longer post to see how the text wrapping behaves within the designated post area. \nIt includes a line break.\nAnd another one here.\nHopefully, it looks okay and doesn\'t overflow or cause strange layout issues on different screen sizes. Testing, testing, one two three.\n>>P'+(now-180000), mediaDataUrl: null, timestamp: now - 50000 }, replies: [] // Example of quoting another reply
             }
        ].map(thread => { // Ensure threadId is set correctly in sample OP posts
            thread.opPost.threadId = thread.id;
            return thread;
        });
    }
    function saveThreads() { /* ... (same as previous step) ... */
        console.log("Saving threads...");
        try {
            const threadsString = JSON.stringify(currentThreads);
            localStorage.setItem(LOCAL_STORAGE_THREADS_KEY, threadsString);
            checkStorageUsage();
            // console.log("Threads saved successfully."); // Reduce noise maybe
            return true;
        } catch (error) {
             console.error("Error saving threads to localStorage:", error);
            if (error.name === 'QuotaExceededError' || (error.code && (error.code === 22 || error.code === 1014))) {
                 const msg = "Error: Local storage limit reached! Cannot save. Clear old threads or browser data.";
                 showStatus(msg, "error", formStatusDiv);
                 showStatus(msg, "error", replyFormStatusDiv);
            } else {
                showStatus("Error saving thread data. See console.", "error", formStatusDiv);
            }
            return false;
        }
    }
    function checkStorageUsage() { /* ... (same as previous step) ... */
        try {
            const item = localStorage.getItem(LOCAL_STORAGE_THREADS_KEY);
            const used = item?.length || 0;
            const usedMB = (used * 2) / (1024 * 1024); // Estimate MB (UTF-16)
            console.log(`Approx. localStorage usage: ${usedMB.toFixed(2)} MB / ${MAX_LOCALSTORAGE_APPROX_MB} MB`);
            if (usedMB > MAX_LOCALSTORAGE_APPROX_MB * 0.8) {
                 console.warn(`LocalStorage usage high (${usedMB.toFixed(2)} MB).`);
            }
        } catch(e) { console.error("Could not check storage usage", e); }
     }
    function filterFoulLanguage(text) { /* ... (same as previous step) ... */
        if (!text || typeof text !== 'string' || BAD_WORDS.length === 0) return text;
        const regex = new RegExp(`\\b(${BAD_WORDS.join('|')})\\b`, 'gi');
        return text.replace(regex, (match) => '*'.repeat(match.length));
     }
    function createPostObject(author, subject, content, mediaDataUrl, parentId = null) { /* ... (same as previous step, includes threadId) ... */
        const timestamp = Date.now();
        const baseId = `${timestamp}-${Math.random().toString(36).substring(2, 7)}`;
        const id = parentId ? `P${baseId}` : `T${baseId}`;
        const threadId = parentId ? parentId : id;

        const filteredAuthor = filterFoulLanguage(author?.trim()) || "Anonymous";
        const filteredSubject = subject ? filterFoulLanguage(subject.trim()) : null;
        const filteredContent = filterFoulLanguage(content?.trim());

        return {
            id: id, parentId: parentId, threadId: threadId, author: filteredAuthor,
            subject: filteredSubject, content: filteredContent,
            mediaDataUrl: mediaDataUrl, timestamp: timestamp
        };
     }

    // --- Actions ---
    function addNewThread(opPost) { /* ... (same as previous step, includes navigation) ... */
        console.log("Adding new thread:", opPost);
        if (!opPost || !opPost.id) {
            console.error("Cannot add thread: Invalid OP post object.");
            showStatus("Internal error creating thread.", "error", formStatusDiv);
            return;
        }
        const newThread = { id: opPost.id, opPost: opPost, replies: [] };
        currentThreads.unshift(newThread);

        if (saveThreads()) {
            window.location.hash = `#thread=${newThread.id}`; // Navigate AFTER save
            // Routing handler will render the page
            showStatus("Thread created.", "success", formStatusDiv);
            if(newThreadForm) newThreadForm.reset(); // Reset form fields visually
            if(threadMediaPreview) clearMediaPreview(threadMediaPreview);
            mediaDataUrlCache.newThread = null;
        } else {
            currentThreads.shift(); // Revert if save failed
            console.error("Failed to save new thread, reverted addition.");
        }
     }
    function addReplyToThread(replyPost) { /* ... (same as previous step) ... */
        console.log("Adding reply:", replyPost);
        if (!replyPost || !replyPost.parentId) {
             console.error("Cannot add reply: Invalid reply object or missing parent ID.");
             showStatus("Internal error posting reply.", "error", replyFormStatusDiv);
            return;
        }
        const threadIndex = currentThreads.findIndex(t => t.id === replyPost.parentId);
        if (threadIndex > -1) {
            currentThreads[threadIndex].replies.push(replyPost);

            if (saveThreads()) {
                 renderSingleThreadPage(replyPost.parentId); // Re-render the current page
                 showStatus("Reply posted.", "success", replyFormStatusDiv);
            } else {
                 currentThreads[threadIndex].replies.pop();
                 console.error("Failed to save reply, reverted addition.");
            }
        } else {
            console.error(`AddReply: Thread with ID ${replyPost.parentId} not found.`);
            showStatus("Error: Parent thread missing.", "error", replyFormStatusDiv);
        }
     }

    // --- Element Creation ---
    function createThreadElement(thread, isThreadPage) { /* ... (same as previous step) ... */
        const threadContainer = document.createElement('div');
        threadContainer.classList.add('thread');
        threadContainer.id = `thread-${thread.id}`;

        if (!thread || !thread.opPost) {
            console.error("Cannot create thread element: Invalid thread data", thread);
            return threadContainer;
        }
        const opElement = createPostElement(thread.opPost, true, isThreadPage);
        if (opElement) threadContainer.appendChild(opElement);

        if (isThreadPage && thread.replies && thread.replies.length > 0) {
            const repliesContainer = document.createElement('div');
            repliesContainer.classList.add('replies-container');
            thread.replies.forEach(reply => {
                if (reply) {
                    const replyElement = createPostElement(reply, false, true);
                    if (replyElement) repliesContainer.appendChild(replyElement);
                } else { console.warn("Skipping invalid reply object:", reply); }
            });
            threadContainer.appendChild(repliesContainer);
        }
        return threadContainer;
    }
    function createPostElement(post, isOP, isThreadPage) { /* ... (same as previous step, includes post-ref-link) ... */
         if (!post || !post.id) {
             console.error("Cannot create post element: Invalid post data", post); return null;
         }
        const postDiv = document.createElement('div');
        postDiv.classList.add('post', isOP ? 'original-post' : 'reply-post');
        postDiv.id = `post-${post.id}`;

        if (isOP && !isThreadPage) {
             postDiv.style.cursor = 'pointer';
             postDiv.addEventListener('click', (e) => {
                 if (!e.target.closest('a, button, img')) { window.location.hash = `#thread=${post.id}`; }
             });
        }

        let mediaHTML = '';
        if (post.mediaDataUrl) {
            if (post.mediaDataUrl.startsWith('data:image/')) { mediaHTML = `<div class="post-media"><img src="${post.mediaDataUrl}" alt="User media" loading="lazy"></div>`; }
            else { mediaHTML = `<div class="post-media"><span class='media-error'>Invalid media data</span></div>`; }
        }

        const contentArea = document.createElement('div');
        contentArea.classList.add('post-content-area');
        const postDate = new Date(post.timestamp || Date.now());
        const formattedDate = postDate.toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' });
        const safeSubject = post.subject ? `<span class="post-subject">${escapeHtml(post.subject)}</span>` : '';
        const safeAuthor = `<span class="post-author">${escapeHtml(post.author)}</span>`;
        const safeBody = escapeHtml(post.content || '[no content]').replace(/\n/g, '<br>');
        const bodyWithRefs = isThreadPage ? safeBody.replace(/>>([TP]\d+-\w+)/g, `<a href="#post-$1" class="post-ref-link" data-ref-id="post-$1">>>$1</a>`) : safeBody;
        const parentLink = !isOP && isThreadPage ? `<a href="#post-${post.parentId}" class="reply-link" title="Parent Post">>>${post.parentId}</a>` : '';
        const postIdSpan = `<span class="post-id" title="Post ID: ${post.id}">No.${post.id.substring(1)}</span>`;
        let headerItems = [safeSubject, safeAuthor, `<span class="post-timestamp" title="${postDate.toISOString()}">${formattedDate}</span>`, postIdSpan, parentLink];

        if (isOP && !isThreadPage) {
            const thread = currentThreads.find(t => t.id === post.id);
            const replyCount = thread?.replies?.length || 0;
            headerItems.push(`<a href="#thread=${post.id}" class="reply-count-indicator" title="View ${replyCount} replies"><i class="fas fa-comments"></i> ${replyCount}</a>`);
        }

        const quoteButtonHTML = isThreadPage ? `<button class="reply-button quote-button" data-post-id="${post.id}" title="Quote this post"><i class="fas fa-quote-right"></i> Quote</button>` : '';

        contentArea.innerHTML = `<div class="post-header">${headerItems.filter(Boolean).join(' ')}</div><div class="post-body">${bodyWithRefs}</div><div class="post-actions">${quoteButtonHTML}</div>`;
        if (mediaHTML) { postDiv.insertAdjacentHTML('afterbegin', mediaHTML); }
        postDiv.appendChild(contentArea);
        return postDiv;
     }

    // --- Event Listeners Setup ---
    function setupEventListeners() { /* ... (same as previous step, includes scroll-to-ref logic) ... */
        console.log("Setting up event listeners...");
        let listenerAttached = false; // Flag to check if *any* listener was attached

        // Helper to safely attach listeners
        const safeAttach = (element, eventType, handler, elementName) => {
            if (element) {
                element.addEventListener(eventType, handler);
                console.log(`Listener attached: ${elementName} ${eventType}`);
                listenerAttached = true; // Mark that at least one listener worked
            } else {
                console.error(`FAILED to attach listener: ${elementName} not found!`);
            }
        };

        // Attach listeners using the helper
        safeAttach(newThreadForm, 'submit', handleFormSubmit('newThread', threadAuthorInput, threadSubjectInput, threadContentInput, addNewThread, formStatusDiv, null), 'newThreadForm');
        safeAttach(replyForm, 'submit', handleFormSubmit('reply', replyAuthorInput, null, replyContentInput, addReplyToThread, replyFormStatusDiv, replyParentIdInput), 'replyForm');
        safeAttach(threadMediaInput, 'change', (e) => handleFileInput(e, threadMediaPreview, 'newThread'), 'threadMediaInput');
        safeAttach(replyMediaInput, 'change', (e) => handleFileInput(e, replyMediaPreview, 'reply'), 'replyMediaInput');
        safeAttach(changeUserBtn, 'click', openUserModal, 'changeUserBtn');
        safeAttach(closeUserModalBtn, 'click', closeUserModal, 'closeUserModalBtn');
        safeAttach(cancelUserBtn, 'click', closeUserModal, 'cancelUserBtn');
        safeAttach(setUserBtn, 'click', handleSetUser, 'setUserBtn');
        safeAttach(userModal, 'click', (e) => { if (e.target === userModal) closeUserModal(); }, 'userModal backdrop');

         // Delegated listeners
         if (singleThreadContainer) {
             singleThreadContainer.addEventListener('click', (event) => {
                 const refLink = event.target.closest('.post-ref-link');
                 const quoteButton = event.target.closest('.quote-button');
                 if (refLink) {
                     event.preventDefault();
                     const targetId = refLink.dataset.refId;
                     if (targetId) scrollToPost(targetId);
                 } else if (quoteButton) {
                     event.preventDefault();
                     const postIdToQuote = quoteButton.dataset.postId;
                     if (postIdToQuote && replyContentInput) {
                         const quoteText = `>>${postIdToQuote}\n`;
                         replyContentInput.value += quoteText;
                         replyContentInput.focus();
                         replyContentInput.selectionStart = replyContentInput.selectionEnd = replyContentInput.value.length;
                         replyFormContainer?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                     }
                 }
             });
             console.log("Listener attached: singleThreadContainer (for post refs & quotes)");
             listenerAttached = true;
         } else { console.error("FAILED to attach listener: singleThreadContainer not found!"); }

         if (pageContentDiv) {
             pageContentDiv.addEventListener('click', (e) => {
                 if (e.target.matches('.back-button') || e.target.closest('.back-button')) {
                     e.preventDefault(); window.location.hash = '#';
                 }
             });
              console.log("Listener attached: pageContentDiv back button delegation");
              listenerAttached = true;
         } else { console.error("FAILED to attach listener: pageContentDiv not found for back button!"); }

        if (!listenerAttached && document.readyState === 'complete') { // Check only after page load is complete
            console.error("MAJOR WARNING: No essential event listeners could be attached. Functionality will be broken. Check HTML element IDs and ensure the script runs after the DOM is ready.");
            alert("Forum Error: UI elements missing. Check console (F12).");
        } else if (listenerAttached) {
            console.log("Event listeners setup complete.");
        }
    }

    // --- Form Handling Logic ---
    function handleFormSubmit(formType, authorInput, subjectInput, contentInput, submitAction, statusDiv, parentIdInput = null) { /* ... (same as previous step) ... */
        return function(event) {
            event.preventDefault();
            if (!statusDiv) { console.error("Status div missing for form", formType); return; }
            showStatus("Processing...", "info", statusDiv);
            const submitButton = event.target.querySelector('button[type="submit"]');
            if (submitButton) submitButton.disabled = true;

            const author = authorInput?.value?.trim() || currentUser;
            const subject = subjectInput?.value;
            const content = contentInput?.value;
            const parentId = parentIdInput?.value;
            const mediaDataUrl = mediaDataUrlCache[formType];

            if (!content || content.trim() === '') {
                showStatus("Comment content is required.", "error", statusDiv);
                if (submitButton) submitButton.disabled = false; return;
            }
            try { /* Storage Check */
                 const currentSize = localStorage.getItem(LOCAL_STORAGE_THREADS_KEY)?.length || 0;
                 const mediaSize = mediaDataUrl?.length || 0;
                 const approxObjectSize = (author.length + (subject?.length || 0) + content.length) * 2;
                 if ((currentSize + mediaSize + approxObjectSize) * 2 > MAX_LOCALSTORAGE_APPROX_MB * 1024 * 1024) {
                     showStatus("Error: Local storage limit likely reached!", "error", statusDiv);
                     if (submitButton) submitButton.disabled = false; return;
                 }
            } catch (e) { console.warn("Could not perform storage check", e); }

            // No setTimeout needed here, direct call is fine
            try {
                const postObject = createPostObject(author, subject, content, mediaDataUrl, parentId);
                mediaDataUrlCache[formType] = null;
                submitAction(postObject);
            } catch (e) {
                 console.error(`Error during ${formType} submission:`, e);
                 showStatus(`An error occurred. See console.`, "error", statusDiv);
            } finally {
                 if (submitButton) submitButton.disabled = false;
            }
        }
    }

    // --- Media Handling ---
    function handleFileInput(event, previewElement, cacheKey) { /* ... (same as previous step, includes stricter size check) ... */
        if (!previewElement) { console.error("Media preview element not found for", cacheKey); return; }
        const file = event.target.files[0];
        clearMediaPreview(previewElement);
        mediaDataUrlCache[cacheKey] = null;
        const statusDiv = cacheKey === 'newThread' ? formStatusDiv : replyFormStatusDiv;
        if (statusDiv) showStatus("", "info", statusDiv);

        if (file) {
            console.log(`File selected (${cacheKey}): ${file.name}, size: ${file.size}, type: ${file.type}`);
            if (!file.type.startsWith('image/png') && !file.type.startsWith('image/jpeg') && !file.type.startsWith('image/gif')) {
                showStatus("Invalid file type (PNG, JPG, GIF only).", "error", statusDiv); event.target.value = ''; return;
            }
            const maxSize = 1 * 1024 * 1024; // 1 MB limit
            if (file.size > maxSize) {
                showStatus(`Image too large (Max ~${(maxSize / (1024*1024)).toFixed(1)}MB)`, "error", statusDiv); event.target.value = ''; return;
            }
            const reader = new FileReader();
            reader.onload = function(e) {
                console.log(`FileReader loaded for ${cacheKey}`);
                if (e.target.result.length > maxSize * 1.5) { // Check encoded size too
                    showStatus(`Image data too large after encoding (Max ~${(maxSize / (1024*1024)).toFixed(1)}MB raw).`, "error", statusDiv);
                    event.target.value = ''; clearMediaPreview(previewElement); return;
                }
                mediaDataUrlCache[cacheKey] = e.target.result;
                const img = document.createElement('img');
                img.src = e.target.result; previewElement.appendChild(img);
            }
            reader.onerror = function(e) {
                console.error("FileReader error:", e); showStatus("Error reading file.", "error", statusDiv);
                mediaDataUrlCache[cacheKey] = null; event.target.value = '';
            }
            reader.readAsDataURL(file);
        }
     }
    function clearMediaPreview(previewElement) { if(previewElement) previewElement.innerHTML = ''; }

    // --- User Switching Modal Handling ---
    function loadCurrentUser() { /* ... (same as previous step) ... */ currentUser = localStorage.getItem(LOCAL_STORAGE_USER_KEY) || 'Anonymous'; console.log("Loaded user:", currentUser); }
    function saveCurrentUser(username) { /* ... (same as previous step) ... */ currentUser = username.trim() || 'Anonymous'; localStorage.setItem(LOCAL_STORAGE_USER_KEY, currentUser); console.log("Saved user:", currentUser); }
    function updateUserDisplay() { /* ... (same as previous step) ... */ if(currentUserDisplay) currentUserDisplay.textContent = escapeHtml(currentUser); }
    function openUserModal() { /* ... (same as previous step) ... */
        if (!userModal || !newUsernameInput) { console.error("User modal elements not found!"); return; }
        newUsernameInput.value = currentUser === 'Anonymous' ? '' : currentUser;
        userModal.classList.remove('hidden'); newUsernameInput.focus(); console.log("User modal opened");
    }
    function closeUserModal() { /* ... (same as previous step) ... */ if(userModal) userModal.classList.add('hidden'); console.log("User modal closed"); }
    function handleSetUser() { /* ... (same as previous step) ... */
        if (!newUsernameInput) return; const newName = newUsernameInput.value;
        saveCurrentUser(newName); updateUserDisplay(); closeUserModal();
     }

    // --- Utilities ---
    function scrollToPost(targetId) { /* ... (same as previous step) ... */
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            console.log("Scrolling to element:", targetId);
            targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            targetElement.classList.add('highlighted-post');
            setTimeout(() => { targetElement.classList.remove('highlighted-post'); }, 1500);
        } else { console.warn(`Target post element not found: ${targetId}`); }
    }
    function showStatus(message, type = "info", element) { /* ... (same as previous step) ... */
         if (!element) return;
        // console.log(`Status (${type}): ${message}`); // Reduce console noise for status
        element.textContent = message;
        element.className = `form-status ${type}`;
        if ((type === 'success' || type === 'info') && message !== "") {
            setTimeout(() => { if (element.textContent === message) { element.textContent = ''; element.className = 'form-status'; } }, 4000);
        }
     }
    function escapeHtml(unsafe) { /* ... (same as previous step) ... */
        if (typeof unsafe !== 'string') return "";
        return unsafe.replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">").replace(/"/g, '"').replace(/'/g, "'");
     }

    // --- Start Forum ---
    initializeForum();

}); // End of DOMContentLoaded