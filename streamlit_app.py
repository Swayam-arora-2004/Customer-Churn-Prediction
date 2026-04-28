import sys
from pathlib import Path

# Add the project root to the Python path
# This ensures that 'src' and 'app' modules can be imported correctly
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the dashboard module to execute the Streamlit app
# We use a try-except to provide a clear error message if dependencies are missing
try:
    import app.dashboard
except Exception as e:
    import streamlit as st
    st.error(f"Failed to load the dashboard: {e}")
    st.info("Make sure all dependencies in requirements.txt are installed.")
    raise e
