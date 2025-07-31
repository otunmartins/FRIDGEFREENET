"""
Streamlit Utilities
===================

Common utility functions for Streamlit UI components.
"""

import streamlit as st

def display_header(title: str, icon: str, description: str) -> None:
    """
    Display a formatted header with title, icon, and description.
    
    Args:
        title: The main title text
        icon: Emoji or icon to display with the title
        description: Descriptive text below the title
    """
    st.markdown(f"# {icon} {title}")
    st.markdown(f"*{description}*")
    st.markdown("---")

def display_summary(text: str) -> None:
    """
    Display summary text in a formatted info box.
    
    Args:
        text: The summary text to display
    """
    st.info(text) 