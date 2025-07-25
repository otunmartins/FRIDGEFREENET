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
    
    // Model selection elements
    modelSelect: document.getElementById('modelSelect'),
    switchModelBtn: document.getElementById('switchModelBtn'),
    currentModelName: document.getElementById('currentModelName'),
    currentModelType: document.getElementById('currentModelType'),
    
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
    loadModelInfo();
    
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
    
    // Model selection
    elements.switchModelBtn?.addEventListener('click', switchModel);
    elements.modelSelect?.addEventListener('change', updateSwitchButton);
    
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
            
            // Handle PSMILES workflow responses with interactive buttons
            if (data.type === 'psmiles_generated_workflow' && data.interactive_buttons) {
                // Add message with workflow content
                let messageContent = data.message;
                
                // Add SVG visualization if available
                if (data.svg_content) {
                    messageContent += `\n\n<div class="svg-container">${data.svg_content}</div>`;
                }
                
                // Add interactive buttons
                messageContent += createInteractiveButtons(data.interactive_buttons);
                
                // Disable typewriter for complex PSMILES content to preserve formatting
                const messageId = addMessage('assistant', messageContent, data.type, null, false);
                
                // Setup button event listeners
                setTimeout(() => {
                    const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
                    if (messageElement) {
                        setupInteractiveButtons(messageElement);
                    }
                }, 500);
            } else if (data.type && data.type.includes('psmiles') && data.interactive_buttons) {
                // Handle other PSMILES responses with structured buttons
                let messageContent = data.message;
                
                if (data.svg_content) {
                    messageContent += `\n\n<div class="svg-container">${data.svg_content}</div>`;
                }
                
                messageContent += createInteractiveButtons(data.interactive_buttons);
                
                // Disable typewriter for complex PSMILES content to preserve formatting
                const messageId = addMessage('assistant', messageContent, data.type, null, false);
                
                setTimeout(() => {
                    const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
                    if (messageElement) {
                        setupInteractiveButtons(messageElement);
                    }
                }, 500);
            } else {
                // Regular message - check if it contains generated HTML buttons or PSMILES content
                // Disable typewriter for PSMILES responses and complex formatted content
                const hasComplexFormatting = data.type && data.type.includes('psmiles');
                const hasHTMLButtons = data.message && (data.message.includes('data-action=') || data.message.includes('<button'));
                const useTypewriter = !hasComplexFormatting && !hasHTMLButtons;
                
                const messageId = addMessage('assistant', data.message, data.type, null, useTypewriter);
                
                // If the message contains HTML buttons, set up their event listeners
                if (hasHTMLButtons) {
                    setTimeout(() => {
                        const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
                        if (messageElement) {
                            setupGeneratedHTMLButtons(messageElement);
                        }
                    }, 500);
                }
            }
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
    if (!elements.chatContainer) return;
    
    // Generate unique ID if not provided
    if (!messageId) {
        messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Determine if typewriter should be used
    const shouldUseTypewriter = useTypewriter !== null ? useTypewriter : (sender === 'assistant');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.id = messageId;
    messageDiv.setAttribute('data-message-id', messageId);
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    
    // Add type indicator if provided
    if (type) {
        messageContent.setAttribute('data-type', type);
        
        // Add special styling based on type
        if (type.includes('psmiles')) {
            messageContent.classList.add('psmiles-response');
        } else if (type === 'literature') {
            messageContent.classList.add('literature-response');
        } else if (type === 'error') {
            messageContent.classList.add('error-response');
        }
    }
    
    messageContent.appendChild(messageBubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    elements.chatContainer.appendChild(messageDiv);
    
    // Handle interactive buttons and SVG content
    let processedContent = content;
    let hasInteractiveButtons = false;
    
    // Check for interactive buttons in content - improved detection
    if (processedContent.includes('workflow-actions') || 
        processedContent.includes('data-action=') || 
        processedContent.includes('<button')) {
        hasInteractiveButtons = true;
    }
    
    // Handle SVG content
    if (processedContent.includes('<div class="svg-container">')) {
        // SVG content is already in the message, just ensure proper styling
        messageContent.classList.add('has-svg-content');
    }
    
    if (shouldUseTypewriter && sender === 'assistant' && !hasInteractiveButtons) {
        // Use typewriter effect for regular assistant messages without buttons
        const typewriter = new TypewriterEffect(messageBubble, processedContent, 3);
        currentTypewriter = typewriter;
        typewriter.start();
    } else {
        // Direct HTML insertion for messages with interactive elements or regular content
        messageBubble.innerHTML = formatMessage(processedContent);
        
        // Setup interactive buttons if present - improved detection and setup
        if (hasInteractiveButtons) {
            // For structured interactive buttons
            if (processedContent.includes('workflow-actions')) {
                setupInteractiveButtons(messageDiv);
            } else if (processedContent.includes('data-action=') || processedContent.includes('<button')) {
                // For LLM-generated HTML buttons, set up event listeners
                setupGeneratedHTMLButtons(messageDiv);
            }
        }
    }
    
    scrollToBottom();
    return messageId;
}

function formatMessage(content) {
    if (!content || typeof content !== 'string') {
        return content;
    }
    
    let formatted = content.trim();
    
    // Handle SVG content specially - don't process markdown for SVG
    if ((formatted.includes('<svg') && formatted.includes('</svg>')) || formatted.includes('<div class="svg-container">')) {
        // Extract SVG container or raw SVG and preserve it
        const svgContainerMatch = formatted.match(/(<div class="svg-container">[\s\S]*?<\/div>)/);
        const svgMatch = formatted.match(/(<svg[\s\S]*?<\/svg>)/);
        
        if (svgContainerMatch) {
            const svgContent = svgContainerMatch[1];
            formatted = formatted.replace(svgContainerMatch[1], `__SVG_CONTENT__`);
            
            // Process the rest as markdown
            formatted = formatted
                .replace(/&quot;/g, '"')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&amp;/g, '&');
            
            // Restore SVG
            formatted = formatted.replace(`__SVG_CONTENT__`, svgContent);
            return formatted;
        } else if (svgMatch) {
            const svgContent = svgMatch[1];
            formatted = formatted.replace(svgMatch[1], `__SVG_CONTENT__`);
            
            // Process the rest as markdown
            formatted = formatted
                .replace(/&quot;/g, '"')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&amp;/g, '&');
            
            // Restore SVG
            formatted = formatted.replace(`__SVG_CONTENT__`, svgContent);
            return formatted;
        }
    }
    
    // Handle HTML content that should not be processed as markdown
    if (formatted.includes('<div') || formatted.includes('<span') || formatted.includes('<button')) {
        // Decode HTML entities but don't process markdown
        formatted = formatted
            .replace(/&quot;/g, '"')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&amp;/g, '&');
        
        // Return as-is to preserve HTML structure
        return formatted;
    }
    
    // Protect HTML elements and code blocks first, BEFORE protecting PSMILES
    const htmlElements = [];
    const codeBlocks = [];
    const inlineCodes = [];
    
    // Extract and temporarily replace code blocks FIRST (they may contain PSMILES)
    formatted = formatted.replace(/```([\s\S]*?)```/g, (match, code) => {
        const placeholder = `__CODEBLOCK_${codeBlocks.length}__`;
        codeBlocks.push(code);
        return placeholder;
    });
    
    // Extract and temporarily replace inline code FIRST (they may contain PSMILES)
    formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
        const placeholder = `__INLINECODE_${inlineCodes.length}__`;
        inlineCodes.push(code);
        return placeholder;
    });
    
    // Protect existing HTML elements from markdown processing
    formatted = formatted.replace(/<(div|span|svg|img|button)[^>]*>.*?<\/\1>/gs, (match) => {
        const placeholder = `__HTML_${htmlElements.length}__`;
        htmlElements.push(match);
        return placeholder;
    });
    
    // NOW protect PSMILES [*] symbols in remaining text (not in code)
    const psmilesList = [];
    formatted = formatted.replace(/\[(\*+)\]/g, (match, stars) => {
        const placeholder = `__PSMILES_${psmilesList.length}__`;
        psmilesList.push(match);
        return placeholder;
    });
    
    // Enhanced line break handling for PSMILES responses - fix the specific formatting issues
    formatted = formatted
        // Add line breaks before headers that are immediately preceded by text (no whitespace)
        .replace(/([^\s\n])(#{1,3}\s)/g, '$1\n\n$2')
        // Add line breaks after headers if they're immediately followed by bold text or content
        .replace(/(#{1,3}[^\n]*?)(\*\*[^:]*?:)/g, '$1\n\n$2')
        // Fix headers followed immediately by other content without line breaks
        .replace(/(#{1,3}[^\n]*?)([A-Z][^#\n]{10,})/g, '$1\n\n$2')
        // Ensure proper spacing around validation sections and lists
        .replace(/(\*\*[^*]+\*\*:?)(\s*)(- \*\*)/g, '$1\n\n$3')
        // Fix bold text followed immediately by headers
        .replace(/(\*\*[^*]+\*\*)\s*(#{1,3}\s)/g, '$1\n\n$2')
        // Handle specific PSMILES format issues - separate sections properly
        .replace(/(EXPLANATION:.*?)(\*\*[^*]+\*\*)/g, '$1\n\n$2')
        .replace(/(characters)\s*(-\s*\*\*)/g, '$1\n\n$2')
        // Ensure list items are properly separated
        .replace(/([^\n])(- \*\*)/g, '$1\n$2');
    
    // Convert markdown to HTML while preserving protected content
    formatted = formatted
        // Headers - process in reverse order to handle ### before ##
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        
        // Bold and italic (now safe since code is protected)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Lists - handle both individual items and wrap in ul tags
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        // Wrap consecutive li elements in ul tags
        .replace(/(<li>.*?<\/li>)(\s*<li>.*?<\/li>)*/gs, '<ul>$&</ul>')
        
        // Line breaks - handle multiple consecutive breaks properly
        .replace(/\n\n+/g, '<br><br>')
        .replace(/\n/g, '<br>');
    
    // Restore HTML elements first
    formatted = formatted.replace(/__HTML_(\d+)__/g, (match, index) => {
        return htmlElements[index];
    });
    
    // Restore code blocks BEFORE PSMILES (so PSMILES in code stay in code)
    formatted = formatted.replace(/__CODEBLOCK_(\d+)__/g, (match, index) => {
        return `<pre><code>${codeBlocks[index]}</code></pre>`;
    });
    
    // Restore inline code BEFORE PSMILES (so PSMILES in code stay in code)
    formatted = formatted.replace(/__INLINECODE_(\d+)__/g, (match, index) => {
        return `<code class="psmiles-code">${inlineCodes[index]}</code>`;
    });
    
    // Finally restore PSMILES [*] symbols (only in non-code text)
    formatted = formatted.replace(/__PSMILES_(\d+)__/g, (match, index) => {
        return `<span class="psmiles-symbol">${psmilesList[index]}</span>`;
    });
    
    return formatted;
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

// ====== Model Management ======
let currentModelInfo = null;
let availableModels = [];

async function loadModelInfo() {
    try {
        const response = await fetch('/api/models/info');
        const data = await response.json();
        
        if (response.ok) {
            currentModelInfo = data.current_model;
            availableModels = data.available_models || [];
            
            updateModelInfo();
            populateModelSelect();
        } else {
            console.error('Failed to load model info:', data.error);
            updateModelInfo({ name: 'Error', type: 'Unknown' });
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        updateModelInfo({ name: 'Connection Error', type: 'Unknown' });
    }
}

function updateModelInfo(modelInfo = null) {
    const info = modelInfo || currentModelInfo || { name: 'Loading...', type: '-' };
    
    if (elements.currentModelName) {
        elements.currentModelName.textContent = info.name || 'Unknown';
        elements.currentModelName.title = info.name || 'Unknown';
    }
    
    if (elements.currentModelType) {
        elements.currentModelType.textContent = info.type || '-';
        elements.currentModelType.title = info.type || '-';
    }
}

function populateModelSelect() {
    if (!elements.modelSelect) return;
    
    // Clear existing options
    elements.modelSelect.innerHTML = '';
    
    if (availableModels.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No models available';
        option.disabled = true;
        elements.modelSelect.appendChild(option);
        return;
    }
    
    // Group models by type
    const modelGroups = {
        ollama: []
    };
    
    // Categorize models by type (LlaSMol removed)
    availableModels.forEach(model => {
        if (model.type === 'ollama') {
            modelGroups.ollama.push(model);
        }
    });
    
    // Add Ollama models
    if (modelGroups.ollama.length > 0) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = 'Ollama Models';
        
        modelGroups.ollama.forEach(model => {
            const option = document.createElement('option');
            option.value = `ollama:${model.name}`;
            option.textContent = `${model.name} (Ollama)`;
            
            // Mark current model as selected
            if (currentModelInfo && 
                currentModelInfo.type === 'ollama' && 
                currentModelInfo.name === model.name) {
                option.selected = true;
            }
            
            optgroup.appendChild(option);
        });
        
        elements.modelSelect.appendChild(optgroup);
    }
    
    // LlaSMol integration removed
    
    updateSwitchButton();
}

function updateSwitchButton() {
    if (!elements.switchModelBtn || !elements.modelSelect) return;
    
    const selectedValue = elements.modelSelect.value;
    const hasSelection = selectedValue && selectedValue !== '';
    
    // Check if selected model is different from current
    let isDifferent = false;
    if (hasSelection && currentModelInfo) {
        const [selectedType, selectedName] = selectedValue.split(':');
        isDifferent = currentModelInfo.type !== selectedType || currentModelInfo.name !== selectedName;
    }
    
    elements.switchModelBtn.disabled = !hasSelection || !isDifferent;
}

async function switchModel() {
    const selectedValue = elements.modelSelect?.value;
    if (!selectedValue) return;
    
    const [modelType, modelName] = selectedValue.split(':');
    
    try {
        // Show loading state
        elements.switchModelBtn.disabled = true;
        elements.switchModelBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        
        const response = await fetch('/api/models/switch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_type: modelType,
                model_name: modelName
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Update current model info
            currentModelInfo = data.current_model;
            updateModelInfo();
            populateModelSelect();
            
            // Show success message
            addMessage('assistant', `✅ Successfully switched to ${modelType.toUpperCase()} model: ${modelName}`, 'success', null, false);
            
            console.log('✅ Model switched successfully:', data.current_model);
        } else {
            throw new Error(data.error || 'Failed to switch model');
        }
    } catch (error) {
        console.error('Error switching model:', error);
        showError(`Failed to switch model: ${error.message}`);
        addMessage('assistant', `❌ Failed to switch model: ${error.message}`, 'error', null, false);
    } finally {
        // Restore button state
        elements.switchModelBtn.innerHTML = '<i class="fas fa-exchange-alt"></i>';
        updateSwitchButton();
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

// ====== Interactive Button Functions ======
function createInteractiveButtons(buttonData) {
    if (!buttonData) return '';
    
    let buttonsHTML = `
    <div class="workflow-actions mt-3">
        <h4>🎯 Choose Your Next Action:</h4>
        <div class="action-categories">
    `;
    
    // Dimerization buttons
    if (buttonData.dimerization) {
        buttonsHTML += `
        <div class="action-category mb-3">
            <h5>${buttonData.dimerization.title}</h5>
            <div class="button-group">
        `;
        
        buttonData.dimerization.buttons.forEach(button => {
            buttonsHTML += `
                <button class="btn btn-${button.style} btn-sm action-btn" 
                        data-action="${button.action}" 
                        data-params='${JSON.stringify(button.params)}'>
                    ${button.label}
                </button>
            `;
        });
        
        buttonsHTML += `
            </div>
        </div>
        `;
    }
    
    // Copolymerization buttons
    if (buttonData.copolymerization) {
        buttonsHTML += `
        <div class="action-category mb-3">
            <h5>${buttonData.copolymerization.title}</h5>
            <div class="button-group">
        `;
        
        buttonData.copolymerization.buttons.forEach(button => {
            if (button.input_fields) {
                // Button with input fields
                buttonsHTML += `
                    <button class="btn btn-${button.style} btn-sm action-btn-input" 
                            data-action="${button.action}" 
                            data-input-fields='${JSON.stringify(button.input_fields)}'>
                        ${button.label}
                    </button>
                `;
            } else {
                // Simple button
                buttonsHTML += `
                    <button class="btn btn-${button.style} btn-sm action-btn" 
                            data-action="${button.action}" 
                            ${button.tooltip ? `title="${button.tooltip}"` : ''}>
                        ${button.label}
                    </button>
                `;
            }
        });
        
        buttonsHTML += `
            </div>
        </div>
        `;
    }
    
    // Addition buttons
    if (buttonData.addition) {
        buttonsHTML += `
        <div class="action-category mb-3">
            <h5>${buttonData.addition.title}</h5>
            <div class="button-group">
        `;
        
        buttonData.addition.buttons.forEach(button => {
            if (button.input_fields) {
                buttonsHTML += `
                    <button class="btn btn-${button.style} btn-sm action-btn-input" 
                            data-action="${button.action}" 
                            data-input-fields='${JSON.stringify(button.input_fields)}'>
                        ${button.label}
                    </button>
                `;
            } else {
                buttonsHTML += `
                    <button class="btn btn-${button.style} btn-sm action-btn" 
                            data-action="${button.action}" 
                            data-params='${JSON.stringify(button.params)}'>
                        ${button.label}
                    </button>
                `;
            }
        });
        
        buttonsHTML += `
            </div>
        </div>
        `;
    }
    
    // Analysis buttons
    if (buttonData.analysis) {
        buttonsHTML += `
        <div class="action-category mb-3">
            <h5>${buttonData.analysis.title}</h5>
            <div class="button-group">
        `;
        
        buttonData.analysis.buttons.forEach(button => {
            buttonsHTML += `
                <button class="btn btn-${button.style} btn-sm action-btn" 
                        data-action="${button.action}" 
                        ${button.tooltip ? `title="${button.tooltip}"` : ''}>
                    ${button.label}
                </button>
            `;
        });
        
        buttonsHTML += `
            </div>
        </div>
        `;
    }
    
    buttonsHTML += `
        </div>
    </div>
    `;
    
    return buttonsHTML;
}

function setupInteractiveButtons(messageElement) {
    // Setup simple action buttons (exclude buttons that already have listeners)
    const actionButtons = messageElement.querySelectorAll('.action-btn:not([data-listener-attached])');
    actionButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            const params = JSON.parse(btn.dataset.params || '{}');
            handleWorkflowAction(action, params);
        });
        // Mark as having listener attached
        btn.setAttribute('data-listener-attached', 'true');
    });
    
    // Setup buttons with input fields (exclude buttons that already have listeners)
    const inputButtons = messageElement.querySelectorAll('.action-btn-input:not([data-listener-attached])');
    inputButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            const inputFields = JSON.parse(btn.dataset.inputFields || '[]');
            console.log('Input button clicked:', action, inputFields); // Debug log
            showInputModal(action, inputFields);
        });
        // Mark as having listener attached
        btn.setAttribute('data-listener-attached', 'true');
    });
}

function setupGeneratedHTMLButtons(messageElement) {
    // Handle buttons generated directly by the LLM with data-action attributes
    // But exclude buttons that are part of the structured workflow (those have specific classes)
    const generatedButtons = messageElement.querySelectorAll('button[data-action]:not(.action-btn):not(.action-btn-input):not([data-listener-attached])');
    generatedButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            let params = {};
            
            // Try to parse params if they exist
            if (btn.dataset.params) {
                try {
                    params = JSON.parse(btn.dataset.params);
                } catch (e) {
                    console.warn('Could not parse button params:', btn.dataset.params);
                }
            }
            
            // Handle common PSMILES actions
            if (action === 'fingerprints') {
                handleWorkflowAction('fingerprints', params);
            } else if (action === 'inchi') {
                handleWorkflowAction('inchi', params);
            } else if (action === 'complete_analysis') {
                handleWorkflowAction('complete_analysis', params);
            } else if (action === 'dimerize') {
                handleWorkflowAction('dimerize', params);
            } else if (action === 'copolymer') {
                handleWorkflowAction('copolymer_input', params);
            } else {
                // Generic action handler
                handleWorkflowAction(action, params);
            }
        });
        
        // Mark as having listener attached
        btn.setAttribute('data-listener-attached', 'true');
        
        // Add visual feedback
        btn.classList.add('btn', 'btn-primary', 'btn-sm');
        if (!btn.classList.contains('generated-btn')) {
            btn.classList.add('generated-btn');
        }
    });
}

function showInputModal(action, inputFields) {
    console.log('showInputModal called with:', action, inputFields); // Debug log
    
    // Create modal HTML
    let modalHTML = `
    <div class="modal fade" id="actionModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${getActionTitle(action)}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="actionForm">
    `;
    
    inputFields.forEach((field, index) => {
        console.log('Processing field:', field); // Debug log
        if (field.type === 'text') {
            modalHTML += `
                <div class="mb-3">
                    <label for="field_${index}" class="form-label">${field.name.replace('_', ' ').toUpperCase()}</label>
                    <input type="text" class="form-control" id="field_${index}" 
                           name="${field.name}" placeholder="${field.placeholder}" required>
                </div>
            `;
        } else if (field.type === 'textarea') {
            modalHTML += `
                <div class="mb-3">
                    <label for="field_${index}" class="form-label">${field.name.replace('_', ' ').toUpperCase()}</label>
                    <textarea class="form-control" id="field_${index}" rows="3"
                              name="${field.name}" placeholder="${field.placeholder}" required></textarea>
                </div>
            `;
        } else if (field.type === 'select') {
            modalHTML += `
                <div class="mb-3">
                    <label for="field_${index}" class="form-label">${field.name.replace('_', ' ').toUpperCase()}</label>
                    <select class="form-control" id="field_${index}" name="${field.name}" required>
            `;
            field.options.forEach(option => {
                modalHTML += `<option value="${option.value}">${option.label}</option>`;
            });
            modalHTML += `
                    </select>
                </div>
            `;
        }
    });
    
    modalHTML += `
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="submitAction">Submit</button>
                </div>
            </div>
        </div>
    </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('actionModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Show modal with proper Bootstrap handling
    const modalElement = document.getElementById('actionModal');
    console.log('Modal element created:', modalElement); // Debug log
    
    let modal;
    try {
        // Check if Bootstrap is available
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            modal = new bootstrap.Modal(modalElement);
        } else if (typeof Modal !== 'undefined') {
            modal = new Modal(modalElement);
        } else {
            // Fallback: show modal manually
            modalElement.style.display = 'block';
            modalElement.classList.add('show');
            document.body.classList.add('modal-open');
        }
        
        if (modal && modal.show) {
            modal.show();
        }
        console.log('Modal shown successfully'); // Debug log
    } catch (error) {
        console.error('Error showing modal:', error);
        // Fallback display
        modalElement.style.display = 'block';
        modalElement.classList.add('show');
        document.body.classList.add('modal-open');
    }
    
    // Setup submit handler
    document.getElementById('submitAction').addEventListener('click', () => {
        console.log('Submit button clicked'); // Debug log
        const form = document.getElementById('actionForm');
        const formData = new FormData(form);
        const params = {};
        
        for (let [key, value] of formData.entries()) {
            params[key] = value;
        }
        
        console.log('Form data collected:', params); // Debug log
        
        // Hide modal
        if (modal && modal.hide) {
            modal.hide();
        } else {
            modalElement.style.display = 'none';
            modalElement.classList.remove('show');
            document.body.classList.remove('modal-open');
            modalElement.remove();
        }
        
        handleWorkflowAction(action, params);
    });
}

function getActionTitle(action) {
    const titles = {
        'copolymer_input': 'Copolymerization Setup',
        'custom_addition': 'Custom Modification'
    };
    return titles[action] || 'Action';
}

async function handleWorkflowAction(action, params) {
    try {
        showLoading();
        
        const response = await fetch('/api/psmiles/action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: action,
                params: params
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Add the result as a new message
            const messageId = 'msg_' + Date.now();
            let messageContent = result.message;
            
            // Add SVG visualization if available
            if (result.svg_content) {
                messageContent += `\n\n### 📊 Structure Visualization\n<div class="svg-container">${result.svg_content}</div>`;
            }
            
            // Add interactive buttons if available
            if (result.interactive_buttons) {
                messageContent += createInteractiveButtons(result.interactive_buttons);
            }
            
            addMessage('assistant', messageContent, result.type, messageId, true);
            
            // Setup button listeners for the new message
            const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
            if (messageElement) {
                setupInteractiveButtons(messageElement);
            }
        } else {
            showError(result.error || 'Action failed');
        }
    } catch (error) {
        console.error('Workflow action error:', error);
        showError('Failed to execute workflow action');
    } finally {
        hideLoading();
    }
}