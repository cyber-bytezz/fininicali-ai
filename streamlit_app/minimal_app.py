"""
Minimal Streamlit App for testing.

This is a simplified version of the application to verify core functionality.
"""
import streamlit as st

# App title
st.title("Financial Market Assistant - Test")
st.write("This is a minimal test application to verify Streamlit is working correctly.")

# Initialize session state
if 'example_query' not in st.session_state:
    st.session_state.example_query = ""

# Define callback for example queries
def set_example_query(query):
    st.session_state.example_query = query
    
# Text input with value from session state
query = st.text_input("Ask a question:", value=st.session_state.example_query)

# Clear the example query after it's been used
if st.session_state.example_query:
    st.session_state.example_query = ""

# Button to submit the query
if query and st.button("Submit"):
    st.write(f"You asked: {query}")
    st.success("Query processed successfully!")

# Example buttons
st.subheader("Try these examples:")
example_queries = [
    "Example query 1",
    "Example query 2",
    "Example query 3"
]

# Display example buttons
for example_query in example_queries:
    if st.button(example_query, key=f"btn_{example_query}", on_click=set_example_query, args=(example_query,)):
        pass  # The callback will handle setting the query
