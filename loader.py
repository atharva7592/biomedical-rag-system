from langchain_community.document_loaders import PyPDFLoader
import os


# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    noise_patterns = [
        "NIH Public Access",
        "Author Manuscript",
        "Publisher",
        "J Pain Symptom Manage",
        "Acknowledgments",
        "Funding for this project",
        "ClinicalTrials.gov",
        "Daly et al.",
        "Page",
    ]

    for line in lines:
        line = line.strip()

        # Remove empty / very short lines
        if len(line) < 5:
            continue

        # Remove figure captions
        if line.startswith("Fig.") or line.startswith("Figure"):
            continue

        # Remove noisy patterns
        if any(pattern in line for pattern in noise_patterns):
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


# -------------------------------
# Load and Clean PDFs
# -------------------------------
def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            for doc in docs:
                doc.page_content = clean_text(doc.page_content)

            documents.extend(docs)

    return documents