import streamlit as st
from rag_core import ask_question

st.set_page_config(page_title="Biomedical Research Assistant")

st.title("Biomedical Research Assistant")
st.write("Ask questions from biomedical research papers.")

query = st.text_input("Enter your question")

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching documents..."):

            answer, sources, chunks = ask_question(query)

        st.subheader("Answer")
        st.write(answer)

        st.write("DEBUG chunks:", chunks)

        st.subheader("Sources")

        for i, source in enumerate(sources):
            st.write(f"Source {i+1}")
            st.write(source)
            st.write("---")

        with st.expander("Retrieved Context"):
            for c in chunks:
                st.write(c)