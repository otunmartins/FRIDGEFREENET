"""
Minimal Design System for Insulin-AI App
Inspired by Apple's design principles with clean, minimal aesthetics
"""

import streamlit as st

def inject_minimal_design_system():
    """
    Inject Apple-inspired minimal design system with day/night mode support
    """
    
    st.markdown("""
    <style>
    /* ===== APPLE-INSPIRED MINIMAL DESIGN SYSTEM ===== */
    
    /* Root Variables for Day/Night Mode */
    :root {
        /* Day Mode Colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #f1f3f4;
        --text-primary: #1d1d1f;
        --text-secondary: #6e6e73;
        --accent-primary: #ff3b30;  /* Streamlit's red accent */
        --accent-secondary: #ff6b6b;
        --border-color: #e5e5e7;
        --shadow-light: rgba(0, 0, 0, 0.04);
        --shadow-medium: rgba(0, 0, 0, 0.08);
    }
    
    /* Night Mode Override */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #000000;
            --bg-secondary: #1c1c1e;
            --bg-tertiary: #2c2c2e;
            --text-primary: #ffffff;
            --text-secondary: #8e8e93;
            --accent-primary: #ff8c42;  /* Orange accent for night mode */
            --accent-secondary: #ffb366;
            --border-color: #38383a;
            --shadow-light: rgba(255, 255, 255, 0.04);
            --shadow-medium: rgba(255, 255, 255, 0.08);
        }
    }
    
    /* Global Reset */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Typography - Clean Apple-style */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.01em;
    }
    
    .subsection-title {
        font-size: 1.125rem;
        font-weight: 500;
        color: var(--text-primary);
        margin: 1.5rem 0 0.75rem 0;
    }
    
    /* Cards - Minimal and Clean */
    .minimal-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px var(--shadow-light);
        transition: all 0.2s ease;
    }
    
    .minimal-card:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 4px 12px var(--shadow-medium);
    }
    
    /* Primary Action Card */
    .primary-card {
        background: var(--bg-primary);
        border: 2px solid var(--accent-primary);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    
    /* Status Indicators - Clean and Simple */
    .status-success {
        background: var(--bg-secondary);
        border-left: 4px solid #34c759;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    .status-warning {
        background: var(--bg-secondary);
        border-left: 4px solid #ff9500;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    .status-error {
        background: var(--bg-secondary);
        border-left: 4px solid var(--accent-primary);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    .status-info {
        background: var(--bg-secondary);
        border-left: 4px solid #007aff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    /* Code Display - Clean Monospace */
    .code-display {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 0.875rem;
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    /* Metrics - Clean and Minimal */
    .metric-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .metric-container:hover {
        border-color: var(--accent-primary);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Progress Indicators */
    .progress-container {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    /* Buttons - Apple Style */
    .stButton > button {
        background: var(--accent-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: var(--accent-secondary) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px var(--shadow-medium) !important;
    }
    
    /* Secondary Buttons */
    .secondary-button {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .secondary-button:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--accent-primary) !important;
    }
    
    /* Navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        color: var(--text-secondary);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .nav-item.active {
        background: var(--accent-primary);
        color: white;
    }
    
    /* Tables - Clean and Minimal */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding: 0.75rem !important;
        font-weight: 500 !important;
    }
    
    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.1) !important;
    }
    
    /* Remove Default Streamlit Styling */
    .css-18e3th9 {
        padding-top: 0;
    }
    
    /* Custom scrollbar for night mode */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .minimal-card {
            padding: 1rem;
        }
        
        .primary-card {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def create_minimal_card(content: str, card_type: str = "default") -> str:
    """Create a minimal card component"""
    card_class = {
        "default": "minimal-card",
        "primary": "primary-card",
        "success": "status-success",
        "warning": "status-warning", 
        "error": "status-error",
        "info": "status-info"
    }.get(card_type, "minimal-card")
    
    return f'<div class="{card_class}">{content}</div>'


def create_metric_card(value: str, label: str) -> str:
    """Create a clean metric display"""
    return f'''
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    '''


def create_section_title(title: str) -> str:
    """Create a clean section title"""
    return f'<h2 class="section-title">{title}</h2>'


def create_subsection_title(title: str) -> str:
    """Create a clean subsection title"""
    return f'<h3 class="subsection-title">{title}</h3>' 