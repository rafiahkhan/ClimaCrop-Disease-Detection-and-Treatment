const API_BASE_URL = window.location.origin;

let selectedFile = null;

// File input handler
document.getElementById('fileInput').addEventListener('change', handleFileSelect);

// Drag and drop handlers
const uploadArea = document.getElementById('uploadArea');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size should be less than 10MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('previewArea').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('previewArea').style.display = 'none';
    hideResults();
}

async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading in left column (below Analyze button)
    document.getElementById('predictionLoading').style.display = 'block';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('predictionCard').style.display = 'none';
    document.getElementById('recommendationsContent').innerHTML = '<p class="recommendations-placeholder">Analyzing image...</p>';

    // Disable analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    try {
        // First, get prediction quickly (without recommendations)
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('include_recommendations', 'false');  // Get prediction first, fast!

        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze image');
        }

        const data = await response.json();
        
        // Display prediction immediately (fast!)
        displayResults(data, false);  // false = no recommendations yet
        
        // Fetch recommendations only for diseased plants; show friendly message for healthy
        if (data.predicted_class) {
            if (isHealthyPrediction(data.predicted_class)) {
                showHealthyMessage();
            } else {
                fetchRecommendations(data.predicted_class);
            }
        }

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while analyzing the image. Please try again.');
    } finally {
        // Re-enable analyze button
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Disease';
    }
}

function displayResults(data, showRecommendations = true) {
    // Hide loading, show prediction card
    document.getElementById('predictionLoading').style.display = 'none';
    document.getElementById('predictionCard').style.display = 'block';

    // Display prediction
    const diseaseName = formatDiseaseName(data.predicted_class);
    document.getElementById('diseaseName').textContent = diseaseName;
    
    const confidence = (data.confidence * 100).toFixed(2);
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    document.getElementById('progressFill').style.width = `${confidence}%`;

    // Display top predictions
    if (data.top_predictions && data.top_predictions.length > 1) {
        const topPredictionsDiv = document.getElementById('topPredictions');
        topPredictionsDiv.innerHTML = '<h3>Other Possible Predictions</h3>';
        
        data.top_predictions.slice(1).forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="class-name">${formatDiseaseName(pred.class)}</span>
                <span class="class-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
            `;
            topPredictionsDiv.appendChild(item);
        });
    } else {
        document.getElementById('topPredictions').innerHTML = '';
    }

    // Display recommendations if available
    const recommendationsDiv = document.getElementById('recommendationsContent');
    if (showRecommendations && data.recommendations) {
        if (data.recommendations_error) {
            recommendationsDiv.innerHTML = `
                <div style="padding: 20px; background: #fef3c7; border-radius: 10px; border-left: 4px solid #f59e0b;">
                    <p><strong>Note:</strong> ${data.recommendations}</p>
                </div>
            `;
        } else {
            recommendationsDiv.innerHTML = formatRecommendations(data.recommendations);
        }
    } else if (!showRecommendations) {
        // Prediction done - now fetching recommendations
        recommendationsDiv.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div class="spinner" style="margin: 0 auto 20px;"></div>
                <p style="color: var(--text-secondary);">Generating treatment recommendations...</p>
            </div>
        `;
    }

    // Scroll to results on mobile
    if (window.innerWidth < 900) {
        document.getElementById('predictionCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function formatRecommendations(text) {
    let formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^### (.*$)/gim, '<h4>$1</h4>')
        .replace(/^## (.*$)/gim, '<h3>$1</h3>')
        .replace(/^# (.*$)/gim, '<h2>$1</h2>')
        .replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>')
        .replace(/^-\s+(.*$)/gim, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    formatted = formatted.split('\n').map(line => {
        if (!line.match(/^<[hul]/) && line.trim()) {
            return `<p>${line}</p>`;
        }
        return line;
    }).join('\n');
    return formatted;
}

function isHealthyPrediction(predictedClass) {
    return /healthy/i.test(predictedClass);
}

function showHealthyMessage() {
    const recommendationsDiv = document.getElementById('recommendationsContent');
    recommendationsDiv.innerHTML = `
        <div class="healthy-message">
            <div class="healthy-icon"><i class="fas fa-check-circle"></i></div>
            <h3>Your plant looks healthy!</h3>
            <p>No treatment needed. Keep up the good work with proper watering, sunlight, and soil care.</p>
            <p class="healthy-tip"><i class="fas fa-leaf"></i> Continue monitoring your plants and maintain good growing conditions to keep them thriving.</p>
        </div>
    `;
}

async function fetchRecommendations(diseaseName) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                disease_name: diseaseName,
                llm_model: 'llama3.1:8b'
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || 'Failed to fetch recommendations');
        }

        const data = await response.json();
        displayRecommendations(data.recommendations);
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        const recommendationsDiv = document.getElementById('recommendationsContent');
        recommendationsDiv.innerHTML = `
            <div style="padding: 20px; background: #fee2e2; border-radius: 10px; border-left: 4px solid #ef4444;">
                <p><strong>Error:</strong> ${error.message || 'Failed to generate recommendations. Please try again later.'}</p>
                <p style="margin-top: 10px; font-size: 0.9em; color: #6b7280;">Make sure Ollama is running: <code>ollama serve</code></p>
            </div>
        `;
    }
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendationsContent');
    recommendationsDiv.innerHTML = formatRecommendations(recommendations);
}

function formatDiseaseName(name) {
    // Convert "Crop_Disease Name" to "Crop - Disease Name"
    return name.replace(/_/g, ' - ').replace(/\b\w/g, l => l.toUpperCase());
}

function showError(message) {
    document.getElementById('predictionLoading').style.display = 'none';
    document.getElementById('predictionCard').style.display = 'none';
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('recommendationsContent').innerHTML = '<p class="recommendations-placeholder">Upload an image and click "Analyze Disease" to see treatment recommendations.</p>';
    
    document.getElementById('errorState').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    document.getElementById('predictionLoading').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('predictionCard').style.display = 'none';
    document.getElementById('recommendationsContent').innerHTML = '<p class="recommendations-placeholder">Upload an image and click "Analyze Disease" to see treatment recommendations.</p>';
}

function resetForm() {
    removeImage();
    hideResults();
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (!data.model_loaded) {
            console.warn('Model not loaded yet');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Run health check on page load
checkHealth();
