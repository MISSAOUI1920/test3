import spacy
import pytextrank
import streamlit as st
import requests
import tarfile
from pathlib import Path
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Function to download the spaCy model
def download_spacy_model():
    url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("en_core_web_sm.tar.gz", "wb") as f:
            f.write(response.raw.read())

        with tarfile.open("en_core_web_sm.tar.gz", "r:gz") as tar:
            tar.extractall()
        
        model_path = Path("en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0")
        return model_path
    else:
        raise Exception("Failed to download spaCy model")

# Check if the model is already downloaded
model_path = Path("en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0")

if not model_path.exists():
    model_path = download_spacy_model()

# Load the spaCy model
nlp = spacy.load(model_path)

# Add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank", last=True)

# Download NLTK data
nltk.download('punkt')

# Streamlit app
st.title("Text Summarization and Key Phrase Extraction")

# Input text
input_text = st.text_area("Enter text to summarize and extract key phrases:", "Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites.[1] Web scraping software may directly access the World Wide Web using the Hypertext Transfer Protocol or a web browser. While web scraping can be done manually by a software user, the term typically refers to automated processes implemented using a bot or web crawler. It is a form of copying in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet, for later retrieval or analysis.")

# Process the input text when the input is not empty
if input_text:
    # Summarize the text
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 4)  # Adjust the number of sentences
    summary_sentences = [str(sentence) for sentence in summary]

    # Display the summary
    st.write("Summary:")
    st.write("\n".join(summary_sentences))

    # Process the summary to extract key phrases
    summary_text = " ".join(summary_sentences)
    doc = nlp(summary_text)
    top_phrases = [phrase.text for phrase in doc._.phrases[:10]]

    # Display the top key phrases
    st.write("Top Key Phrases:")
    st.write("\n".join(top_phrases))
