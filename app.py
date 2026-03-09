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

        for i, s in enumerate(sources):

            source_file = s.get("source", "Unknown document")
            page = s.get("page", "Unknown page")

            st.write(f"Source {i+1}")
            st.write(f"Document: {source_file}")
            st.write(f"Page: {page}")
            st.write("---")

            with st.expander("Retrieved Context"):
                for c in chunks:
                    st.write(c)