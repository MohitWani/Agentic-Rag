import streamlit as st
from agent import graph_workflow

st.title("Ask me the query about HDFC MF Factsheet")

graph = graph_workflow()

query = st.text_input("Enter Your Query.")


inputs = {
        "messages": [
            ("user", query),
        ]
    }

if query:
    for output in graph.stream(inputs):
        for key, value in output.items():
            st.write(f"Output from node '{key}':")
            st.write("---")
            st.write(value['messages'], indent=2, width=80, depth=None)
        st.write("\n---\n")
