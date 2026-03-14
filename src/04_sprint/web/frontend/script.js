document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const queryInput = document.getElementById('query-input');
    const annotatedCheckbox = document.getElementById('annotated-only');
    const labelFilter = document.getElementById('label-filter');
    const resultsArea = document.getElementById('results-area');

    // Use a local FastAPI server for development
    const API_BASE_URL = 'http://127.0.0.1:8000';

    // Fetch initial corpus stats
    fetchCorpusStats();

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = queryInput.value.trim();
        if (!query) return;

        const isAnnotatedOnly = annotatedCheckbox.checked;
        const selectedLabel = labelFilter ? labelFilter.value : '';

        // Show loading state
        resultsArea.innerHTML = '<div class="loading">Searching corpus...</div>';

        try {
            let searchUrl = `${API_BASE_URL}/search?q=${encodeURIComponent(query)}&annotated_only=${isAnnotatedOnly}`;
            if (selectedLabel) {
                searchUrl += `&label=${encodeURIComponent(selectedLabel)}`;
            }

            const response = await fetch(searchUrl);

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            renderResults(data, query);

        } catch (error) {
            console.error('Search error:', error);
            resultsArea.innerHTML = `<div class="placeholder-text" style="color: #dc3545;">
                <p>Error connecting to search server.</p>
                <p style="font-size: 0.85rem; margin-top: 0.5rem">Make sure the FastAPI backend is running on ${API_BASE_URL}</p>
            </div>`;
        }
    });

    function renderResults(data, query) {
        if (!data.results || data.results.length === 0) {
            resultsArea.innerHTML = `<div class="placeholder-text"><p>No relevant texts found for "${query}".</p></div>`;
            return;
        }

        const html = [
            `<div style="margin-bottom: 1.5rem; font-size: 0.9rem; color: #6c757d;">
                Found ${data.total_hits} results in ${data.search_time_ms}ms
            </div>`
        ];

        data.results.forEach(item => {
            // Whoosh now returns raw HTML syntax directly in snippet text with `<span class="highlight">...</span>`

            const badgeHtml = item.is_annotated
                ? '<span class="badge badge-annotated">✓ Annotated</span>'
                : '<span class="badge">Raw Text</span>';
                
            const labelBadge = `<span class="badge" style="background-color: #e2e3e5; margin-right: 8px;">${item.label.toUpperCase()}</span>`;

            html.push(`
                <div class="result-item">
                    <div class="doc-id">Document ID: ${item.doc_id} ${labelBadge}${badgeHtml}</div>
                    <div class="snippet">...${item.snippet || item.snippet_zh}...</div>
                </div>
            `);
        });

        resultsArea.innerHTML = html.join('');
    }

    async function fetchCorpusStats() {
        try {
            const response = await fetch(`${API_BASE_URL}/stats`);
            if (!response.ok) return;
            
            const stats = await response.json();
            
            // Update counts in the UI
            updateStatElement('stat-total-docs', stats.total_docs);
            updateStatElement('stat-annotated-docs', stats.annotated_docs);
            updateStatElement('stat-raw-docs', stats.raw_docs);
            
            updateStatElement('stat-label-ham', stats.labels?.ham || 0);
            updateStatElement('stat-label-spam', stats.labels?.spam || 0);
            updateStatElement('stat-label-phish', stats.labels?.phish || 0);
            
        } catch (error) {
            console.error('Failed to fetch corpus stats:', error);
        }
    }

    function updateStatElement(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = `(${value.toLocaleString()} Docs)`;
        }
    }

    const activeSearchOptions = new Set();
    let currentMetadataFilter = { text: '', value: '' };
    const pillsContainer = document.getElementById('search-pills-container');

    function createPillElement(text, onRemove) {
        const pill = document.createElement('div');
        pill.className = 'search-pill';
        
        const textSpan = document.createElement('span');
        textSpan.textContent = text;
        
        const closeBtn = document.createElement('span');
        closeBtn.className = 'remove-pill';
        closeBtn.innerHTML = '&times;';
        closeBtn.setAttribute('aria-label', 'Remove option');
        closeBtn.title = 'Remove';
        
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            onRemove();
            renderSearchPills();
            queryInput.focus();
        });
        
        pill.appendChild(textSpan);
        pill.appendChild(closeBtn);
        return pill;
    }

    function renderSearchPills() {
        if (!pillsContainer) return;
        pillsContainer.innerHTML = '';
        
        if (currentMetadataFilter.value && currentMetadataFilter.text) {
            const pill = createPillElement(currentMetadataFilter.text, () => {
                currentMetadataFilter = { text: '', value: '' };
                if (labelFilter) labelFilter.value = '';
            });
            pillsContainer.appendChild(pill);
        }
        
        activeSearchOptions.forEach(optionText => {
            const pill = createPillElement(optionText, () => {
                activeSearchOptions.delete(optionText);
            });
            pillsContainer.appendChild(pill);
        });
    }

    if (labelFilter) {
        labelFilter.addEventListener('change', () => {
            if (labelFilter.value) {
                const optionText = labelFilter.options[labelFilter.selectedIndex].text;
                currentMetadataFilter = { value: labelFilter.value, text: optionText };
            } else {
                currentMetadataFilter = { text: '', value: '' };
            }
            renderSearchPills();
        });
    }

    // Toggle selected search option in the search bar
    document.querySelectorAll('.search-option-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const optionText = link.textContent.replace(/\s+/g, ' ').trim();
            
            if (activeSearchOptions.has(optionText)) {
                activeSearchOptions.delete(optionText);
            } else {
                activeSearchOptions.add(optionText);
            }
            
            renderSearchPills();
            queryInput.focus();
        });
    });

    // Prevent default jump for placeholder links (for other links without specific behavior)
    document.querySelectorAll('a[href="#"]:not(.search-option-link)').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Optional: you could show a quick toast/alert here like:
            // alert("This is a placeholder in the initial mockup UI.");
        });
    });
});
