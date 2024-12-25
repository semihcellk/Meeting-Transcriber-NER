import streamlit as st
from transformers import pipeline

# Load the Whisper Model
def load_whisper_model():
    # Load the Whisper model for automatic speech recognition
    pipe1 = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return pipe1  # Return the loaded model

# Load the NER (Named Entity Recognition) Model
def load_ner_model():
    # Load a token classification pipeline for NER with the "dslim/bert-base-NER" model
    # Using aggregation_strategy="simple" to combine fragmented entities
    pipe2 = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return pipe2  # Return the loaded model

# Transcribe audio into text using Whisper model
def transcribe_audio(uploaded_file):
    whisper_model = load_whisper_model()  # Load the Whisper model
    read_file = uploaded_file.read()  # Read the uploaded audio file as binary data
    text = whisper_model(read_file, return_timestamps=True)  # Transcribe the audio
    return text["text"]  # Return the transcribed text

# Extract entities (Organizations, Locations, Persons) from the transcribed text
def extract_entities(text, ner_pipeline):
    ner_model = ner_pipeline(text)  # Perform NER on the transcribed text
    entities = {"ORGs": [], "LOCs": [], "PERs": []}  # Initialize a dictionary for storing entities

    # Loop through each recognized entity from the model
    for result in ner_model:
        entity_type = result["entity_group"]  # Get the type of the entity (ORG, LOC, PER, etc.)
        entity_word = result["word"]  # Get the recognized word

        # Normalize the entity type by removing "B-" or "I-" prefixes
        if entity_type.startswith("B-") or entity_type.startswith("I-"):
            entity_type = entity_type[2:]

        # Add the entity to the corresponding category (ORG, LOC, PER)
        if entity_type == "ORG":
            entities["ORGs"].append(entity_word)
        elif entity_type == "LOC":
            entities["LOCs"].append(entity_word)
        elif entity_type == "PER":
            entities["PERs"].append(entity_word)

    # Remove duplicate entities while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
        
    return entities  # Return the extracted entities

# Main Streamlit app
def main():
    st.title("Meeting Transcription and Entity Extraction")  # Display the app title

    # Display student information
    STUDENT_NAME = "Semih Çelik"
    STUDENT_ID = "150230104"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")
    
    # Provide instructions for the app
    st.write("""
    Upload a business meeting audio file to:

    1. Transcribe the meeting audio into text.
    2. Extract key entities such as Person, Organizations, Dates, and Locations.
    """)

    # File uploader for uploading an audio file (only accepts WAV format)
    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    if uploaded_file is not None:
        # Transcribe the uploaded audio file
        st.info("Transcribing the audio file... This may take a minute.")
        text = transcribe_audio(uploaded_file)
        st.subheader("Transcription:")  # Display transcription title
        st.write(text)  # Show the transcribed text

        # Perform Named Entity Recognition (NER) on the transcribed text
        st.info("Extracting entities...")
        ner_pipeline = load_ner_model()  # Load the NER model
        extract = extract_entities(text, ner_pipeline)  # Extract entities

        # Display extracted entities in separate sections
        st.subheader("Extracted Entities:")
        st.subheader("Organizations (ORGs):")
        st.markdown("- " + "\n- ".join(extract["ORGs"]))  # Display organizations as a markdown list
        st.subheader("Locations (LOCs):")
        st.markdown("- " + "\n- ".join(extract["LOCs"]))  # Display locations as a markdown list
        st.subheader("Persons (PERs):")
        st.markdown("- " + "\n- ".join(extract["PERs"]))  # Display persons as a markdown list

# Entry point for the app
if __name__ == "__main__":
    main()
