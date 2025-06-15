/**
 * Insulin AI Web Application
 * Frontend JavaScript for ChatGPT-like interface
 */

// ====== Global Variables ======
let currentMode = 'general';
let isLoading = false;
let examples = {};

// ====== DOM Elements ======
const elements = {
    // Sidebar elements
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    newChatBtn: document.getElementById('newChatBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    modeButtons: document.querySelectorAll('.mode-btn'),
    currentMode: document.getElementById('currentMode'),
    modeDescription: document.getElementById('modeDescription'),
    examplesList: document.getElementById('examplesList'),
    
    // Chat elements
    chatContainer: document.getElementById('chatContainer'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    charCount: document.getElementById('charCount'),
    
    // Status elements
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    
    // Modal elements
    errorModal: document.getElementById('errorModal'),
    errorMessage: document.getElementById('errorMessage')
};

// ====== Mode Configurations ======
const modeConfig = {
    general: {
        title: 'General Chat',
        description: 'Ask questions about the insulin AI project',
        placeholder: 'Ask about insulin delivery, materials science, or project details...'
    },
    literature: {
        title: 'Literature Mining',
        description: 'Advanced literature mining with AI-powered analysis',
        placeholder: 'Search for materials, papers, or research topics...'
    },
    psmiles: {
        title: 'PSMILES Generator',
        description: 'Generate and validate Polymer SMILES strings',
        placeholder: 'Describe a polymer structure, ask for examples, or validate a PSMILES...'
    }
};

// ====== Typewriter Effect Functions ======
class TypewriterEffect {
    constructor(element, text, speed = 5) {
        this.element = element;
        this.text = text;
        this.speed = speed;
        this.index = 0;
        this.isTyping = false;
    }
    
    async start() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        this.element.innerHTML = '';
        
        // Format the markdown text to HTML first
        const formattedHTML = formatMessage(this.text);
        
        // Create a temporary div to extract plain text content
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = formattedHTML;
        const plainText = tempDiv.textContent || tempDiv.innerText || '';
        
        // Type out character by character, but apply full formatting at the end
        for (let i = 0; i <= plainText.length; i++) {
            if (!this.isTyping) break;
            
            // Show partial text with cursor
            const partialText = plainText.substring(0, i);
            this.element.innerHTML = this.formatPartialText(partialText) + '<span class="typewriter-cursor"></span>';
            
            await this.delay(this.speed);
        }
        
        // Show final formatted content without cursor
        if (this.isTyping) {
            this.element.innerHTML = formattedHTML;
        }
        
        this.isTyping = false;
    }
    
    formatPartialText(text) {
        // Simple formatting for partial text (without breaking HTML)
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }
    
    stop() {
        this.isTyping = false;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    parseMarkdown(text) {
        // Handle markdown formatting while preserving structure for typewriter effect
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
    }
}

// Global typewriter instance
let currentTypewriter = null;

// ====== Initialization ======
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🚀 Initializing Insulin AI Web App...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial data
    loadExamples();
    checkSystemStatus();
    
    // Setup auto-resize for textarea
    setupTextareaAutoResize();
    
    // Set initial mode
    setMode('general');
    
    console.log('✅ App initialized successfully!');
}

// ====== Event Listeners ======
function setupEventListeners() {
    // Sidebar toggle (mobile)
    elements.sidebarToggle?.addEventListener('click', toggleSidebar);
    
    // New chat button
    elements.newChatBtn?.addEventListener('click', startNewChat);
    
    // Clear chat button
    elements.clearChatBtn?.addEventListener('click', clearChat);
    
    // Mode buttons
    elements.modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            setMode(mode);
        });
    });
    
    // Message input
    elements.messageInput?.addEventListener('input', handleInputChange);
    elements.messageInput?.addEventListener('keydown', handleKeyDown);
    
    // Send button
    elements.sendBtn?.addEventListener('click', sendMessage);
    
    // Close modal on outside click
    elements.errorModal?.addEventListener('click', (e) => {
        if (e.target === elements.errorModal) {
            closeModal('errorModal');
        }
    });
    
    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            elements.sidebar?.classList.contains('open') &&
            !elements.sidebar.contains(e.target) &&
            !elements.sidebarToggle?.contains(e.target)) {
            closeSidebar();
        }
    });
    
    // Window resize handler
    window.addEventListener('resize', handleWindowResize);
}

// ====== Chat Functionality ======
async function sendMessage() {
    const message = elements.messageInput?.value.trim();
    if (!message || isLoading) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Clear input
    elements.messageInput.value = '';
    updateCharCount();
    updateSendButton();
    
    try {
        if (currentMode === 'literature') {
            // Use literature mining with real-time progress
            await handleLiteratureStreaming(message);
        } else {
            // Use regular API for general chat
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    type: currentMode
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'An error occurred');
            }
            
            // Add AI response to chat
            addMessage('assistant', data.message, data.type, null, true);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`, 'error', null, true);
        showError(error.message);
    }
}

async function handleLiteratureStreaming(message) {
    // Create a progress message element that we'll update in real-time
    const progressMessageId = 'progress-' + Date.now();
    const progressElement = addMessage('assistant', 'Starting literature mining...', 'literature', progressMessageId, false);
    
    // LOG: Confirm timeout removal
    console.log('🚀 Literature mining started - NO TIMEOUT LIMITS');
    
    try {
        // URL encode the message for the GET request
        const encodedMessage = encodeURIComponent(message);
        
        // Create EventSource for Server-Sent Events with the message in the URL
        const eventSource = new EventSource(`/api/literature-stream/${encodedMessage}`);
        
        let progressContent = '';
        let hasStarted = false;
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                // Skip heartbeat messages
                if (data.type === 'heartbeat') {
                    return;
                }
                
                if (data.type === 'start') {
                    progressContent = '🚀 **Starting Literature Mining Process**\n\n';
                    hasStarted = true;
                } else if (data.type === 'progress') {
                    if (data.step_type === 'explanation') {
                        progressContent += `💭 **AI Thinking:** ${data.message}\n\n`;
                    } else if (data.step_type === 'start') {
                        progressContent += `🎯 **${data.message}**\n\n`;
                    } else if (data.step_type === 'complete') {
                        progressContent += `✅ **${data.message}**\n\n`;
                    } else {
                        progressContent += `${data.message}\n\n`;
                    }
                } else if (data.type === 'results') {
                    // Final results - replace the progress message
                    updateMessage(progressMessageId, data.message);
                    eventSource.close();
                    return;
                } else if (data.type === 'error') {
                    progressContent += `❌ **Error:** ${data.message}\n\n`;
                    updateMessage(progressMessageId, progressContent, false);
                    eventSource.close();
                    return;
                }
                
                // Update the progress message in real-time (immediate display)
                if (hasStarted) {
                    const messageBubble = progressElement.querySelector('.message-bubble');
                    if (messageBubble) {
                        messageBubble.innerHTML = formatMessage(progressContent);
                    }
                    
                    // Auto-scroll to bottom to follow progress
                    scrollToBottom();
                }
                
            } catch (e) {
                console.error('Error parsing SSE data:', e);
            }
        };
        
        eventSource.onerror = function(event) {
            console.error('EventSource error:', event);
            progressContent += `❌ **Connection Error:** Lost connection to server\n\n`;
            updateMessage(progressMessageId, progressContent, false);
            eventSource.close();
        };
        
        // TIMEOUT REMOVED: No time limit - let literature mining complete naturally
        
    } catch (error) {
        console.error('Literature streaming error:', error);
        updateMessage(progressMessageId, `❌ Error during literature mining: ${error.message}`, false);
        throw error;
    }
}

// Helper function to update an existing message
function updateMessage(messageId, newContent, useTypewriter = true) {
    const messageElement = document.getElementById(messageId);
    if (messageElement) {
        const messageBubble = messageElement.querySelector('.message-bubble');
        if (messageBubble) {
            // Use typewriter for final results but not for real-time progress updates
            if (useTypewriter) {
                currentTypewriter = new TypewriterEffect(messageBubble, newContent);
                currentTypewriter.start();
            } else {
                messageBubble.innerHTML = formatMessage(newContent);
            }
        }
    }
}

// Helper function to scroll chat to bottom
function scrollToBottom() {
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

function addMessage(sender, content, type = null, messageId = null, useTypewriter = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender} fade-in`;
    
    // Set ID if provided (for real-time updates)
    if (messageId) {
        messageDiv.id = messageId;
    }
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    
    // Format message content
    messageBubble.innerHTML = formatMessage(content);
    
    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString();
    
    messageContent.appendChild(messageBubble);
    messageContent.appendChild(messageTime);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    // Remove welcome message if it exists
    const welcomeMessage = elements.chatContainer?.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    elements.chatContainer?.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
    
    // Use fast typewriter effect for ALL assistant messages by default
    // Only skip if explicitly set to false
    if (sender === 'assistant' && useTypewriter !== false) {
        currentTypewriter = new TypewriterEffect(messageBubble, content);
        currentTypewriter.start();
    }
    
    return messageDiv; // Return the element for further manipulation
}

// Helper function to format message content (markdown and other formatting)
function formatMessage(content) {
    // Skip placeholder system entirely and format directly
    let result = content;
    
    // Step 1: Clean up any placeholder artifacts and HTML issues FIRST
    result = result.replace(/CHEMICAL_PLACEHOLDER_\d+/g, 'PSMILES');
    result = result.replace(/__CHEMICAL_PLACEHOLDER_\d+__/g, '[*]');
    
    // Fix HTML artifact issues - remove malformed HTML before headers
    result = result.replace(/[a-zA-Z-]+">([#]*\s*[🧪🔍📚🎯✅❌⚠️💡🚀]\s*)/g, '$1');
    result = result.replace(/PSMILES-generation-results">/g, '');
    result = result.replace(/psmiles-[a-z-]+\">/g, '');
    
    // Remove redundant PSMILES lines in explanations
    result = result.replace(/AI Explanation:\s*PSMILES:\s*\[[^\]]*\][A-Za-z0-9\[\]\(\)=\#\-\+\*]*\s*\n\s*/g, 'AI Explanation:\n');
    result = result.replace(/PSMILES:\s*\[[^\]]*\][A-Za-z0-9\[\]\(\)=\#\-\+\*]*\s*\n\s*EXPLANATION:/g, 'EXPLANATION:');
    
    // Step 2: Process with markdown
    if (typeof marked !== 'undefined') {
        // Configure marked to be more lenient with HTML
        marked.setOptions({
            breaks: true,
            gfm: true,
            sanitize: false,
            pedantic: false,
            smartLists: true,
            smartypants: false
        });
        
        result = marked.parse(result);
    } else {
        // Fallback processing
        result = result
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            .replace(/^(\d+\.)\s(.*$)/gm, '<ol><li>$2</li></ol>')
            .replace(/---/g, '<hr>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^(.)/gm, '<p>$1')
            .replace(/<p><\/p>/g, '')
            + '</p>';
    }
    
    // Step 3: Apply chemical formatting directly after markdown
    // Format PSMILES: patterns
    result = result.replace(/PSMILES:\s*([A-Za-z0-9\[\]\(\)=\#\-\+\*]+)/gi, 
        'PSMILES: <code class="psmiles-code">$1</code>');
    
    // Format standalone chemical symbols [*] and elements
    result = result.replace(/\[\*\]/g, '<span class="chemical-symbol">[*]</span>');
    result = result.replace(/\[([A-Z][a-z]?)\]/g, '<span class="chemical-symbol">[$1]</span>');
    
    // Format the word PSMILES itself (not in code blocks)
    result = result.replace(/(?<!<code[^>]*>.*?)\bPSMILES\b(?!.*?<\/code>)/gi, 
        '<span class="psmiles-keyword">PSMILES</span>');
    
    // Format chemical formulas
    result = result.replace(/\b([A-Z][a-z]?\d+[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b/g, 
        '<span class="chemical-formula">$1</span>');
    
    // Step 4: Final cleanup of any remaining artifacts
    result = result.replace(/class="[^"]*-generation-results[^"]*"/g, '');
    
    return result;
}

function clearChat() {
    if (confirm('Are you sure you want to clear the conversation?')) {
        // Remove all messages
        const messages = elements.chatContainer?.querySelectorAll('.message');
        messages?.forEach(msg => msg.remove());
        
        // Show welcome message
        showWelcomeMessage();
        
        // Call API to clear server-side history
        fetch('/api/new-chat', { method: 'POST' })
            .catch(error => console.error('Error clearing chat:', error));
    }
}

function showWelcomeMessage() {
    const welcomeHTML = `
        <div class="welcome-message">
            <div class="welcome-content">
                <div class="welcome-icon">
                    <i class="fas fa-microscope"></i>
                </div>
                <h2>Welcome to Insulin AI</h2>
                <p>An intelligent assistant for discovering fridge-free insulin delivery materials</p>
                <div class="welcome-features">
                    <div class="feature">
                        <i class="fas fa-book"></i>
                        <span>Literature Mining</span>
                    </div>
                    <div class="feature">
                        <i class="fas fa-atom"></i>
                        <span>Material Discovery</span>
                    </div>
                    <div class="feature">
                        <i class="fas fa-brain"></i>
                        <span>AI-Powered Research</span>
                    </div>
                </div>
                <p class="welcome-instruction">Choose a chat mode and start exploring!</p>
            </div>
        </div>
    `;
    elements.chatContainer.innerHTML = welcomeHTML;
}

async function startNewChat() {
    try {
        const response = await fetch('/api/new-chat', { method: 'POST' });
        const data = await response.json();
        
        if (response.ok) {
            clearChatMessages();
            showWelcomeMessage();
        } else {
            throw new Error(data.error || 'Failed to start new chat');
        }
    } catch (error) {
        console.error('Error starting new chat:', error);
        showError(error.message);
    }
}

function clearChatMessages() {
    const messages = elements.chatContainer?.querySelectorAll('.message');
    messages?.forEach(msg => msg.remove());
}

// ====== Mode Management ======
function setMode(mode) {
    if (!modeConfig[mode]) return;
    
    currentMode = mode;
    const config = modeConfig[mode];
    
    // Update mode buttons
    elements.modeButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    // Update header
    if (elements.currentMode) {
        elements.currentMode.textContent = config.title;
    }
    if (elements.modeDescription) {
        elements.modeDescription.textContent = config.description;
    }
    
    // Update input placeholder
    if (elements.messageInput) {
        elements.messageInput.placeholder = config.placeholder;
    }
    
    // Update examples
    updateExamples(mode);
    
    console.log(`📱 Switched to ${mode} mode`);
}

// ====== Input Handling ======
function handleInputChange() {
    updateCharCount();
    updateSendButton();
    autoResizeTextarea();
}

function handleKeyDown(e) {
    if (e.key === 'Enter') {
        if (e.shiftKey) {
            // Allow new line
            return;
        } else {
            // Send message
            e.preventDefault();
            sendMessage();
        }
    }
}

function updateCharCount() {
    const count = elements.messageInput?.value.length || 0;
    if (elements.charCount) {
        elements.charCount.textContent = count;
    }
}

function updateSendButton() {
    const hasContent = elements.messageInput?.value.trim().length > 0;
    if (elements.sendBtn) {
        elements.sendBtn.disabled = !hasContent || isLoading;
    }
}

function setupTextareaAutoResize() {
    if (!elements.messageInput) return;
    
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.overflowY = 'hidden';
}

function autoResizeTextarea() {
    if (!elements.messageInput) return;
    
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
}

// ====== Examples Management ======
async function loadExamples() {
    try {
        const response = await fetch('/api/examples');
        const data = await response.json();
        
        if (response.ok) {
            examples = data;
            updateExamples(currentMode);
        } else {
            console.error('Failed to load examples:', data.error);
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

function updateExamples(mode) {
    if (!elements.examplesList || !examples[mode]) return;
    
    elements.examplesList.innerHTML = '';
    
    examples[mode].forEach(example => {
        const exampleDiv = document.createElement('div');
        exampleDiv.className = 'example-item';
        exampleDiv.textContent = example;
        exampleDiv.addEventListener('click', () => {
            elements.messageInput.value = example;
            updateCharCount();
            updateSendButton();
            elements.messageInput.focus();
        });
        elements.examplesList.appendChild(exampleDiv);
    });
}

// ====== Status Management ======
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (response.ok) {
            updateStatusIndicator(data);
        } else {
            updateStatusIndicator({ 
                literature_miner: false, 
                chatbot: false, 
                ollama_connection: 'Error' 
            });
        }
    } catch (error) {
        console.error('Error checking status:', error);
        updateStatusIndicator({ 
            literature_miner: false, 
            chatbot: false, 
            ollama_connection: 'Connection failed' 
        });
    }
}

function updateStatusIndicator(status) {
    const isOnline = status.literature_miner && status.chatbot && 
                    typeof status.ollama_connection === 'string' && 
                    status.ollama_connection.includes('Connected');
    
    if (elements.statusDot) {
        elements.statusDot.className = 'status-dot ' + (isOnline ? 'online' : 'offline');
    }
    
    if (elements.statusText) {
        elements.statusText.textContent = isOnline ? 'All systems online' : 'System offline';
    }
}

// ====== Loading States ======
function showLoading() {
    // Loading overlay removed - AI agent works directly
    isLoading = true;
    updateSendButton();
}

function hideLoading() {
    // Loading overlay removed - AI agent works directly  
    isLoading = false;
    updateSendButton();
}

// ====== Modal Management ======
function showError(message) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
    if (elements.errorModal) {
        elements.errorModal.classList.add('active');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// Make closeModal globally available for onclick handlers
window.closeModal = closeModal;

// ====== Sidebar Management ======
function toggleSidebar() {
    if (elements.sidebar) {
        elements.sidebar.classList.toggle('open');
    }
}

function closeSidebar() {
    if (elements.sidebar) {
        elements.sidebar.classList.remove('open');
    }
}

// ====== Responsive Handling ======
function handleWindowResize() {
    // Close sidebar on desktop
    if (window.innerWidth > 768) {
        closeSidebar();
    }
    
    // Auto-resize textarea
    autoResizeTextarea();
}

// ====== Utility Functions ======
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ====== Keyboard Shortcuts ======
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        closeModal('errorModal');
        closeSidebar();
    }
    
    // Focus message input with '/'
    if (e.key === '/' && e.target !== elements.messageInput) {
        e.preventDefault();
        elements.messageInput?.focus();
    }
});

// ====== Auto-refresh Status ======
setInterval(checkSystemStatus, 30000); // Check every 30 seconds

// ====== Service Worker (for future PWA features) ======
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Future: register service worker for offline functionality
        console.log('🔄 Service Worker support detected');
    });
}

// ====== Error Handling ======
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});

console.log('📱 Insulin AI Frontend loaded successfully!'); 