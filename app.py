import streamlit as st
from transformers import pipeline

def load_whisper_model():
    
    pipe1 = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    return pipe1

def load_ner_model():
    
    pipe2 = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    return pipe2

def transcribe_audio(uploaded_file):
    whisper_model = load_whisper_model()
    read_file = uploaded_file.read()
    text = whisper_model(read_file, return_timestamps=True)
    return text["text"]

def extract_entities(text, ner_pipeline):
    ner_model = ner_pipeline(text)
    entities = {"ORGs": [], "LOCs": [], "PERs": []}

    for result in ner_model:
        entity_type = result["entity_group"]
        entity_word = result["word"]

        if entity_type.startswith("B-") or entity_type.startswith("I-"):
            entity_type = entity_type[2:]

        if entity_type == "ORG":
            entities["ORGs"].append(entity_word)
        elif entity_type == "LOC":
            entities["LOCs"].append(entity_word)
        elif entity_type == "PER":
            entities["PERs"].append(entity_word)
            
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
        
    return entities

def main():
    st.title("Meeting Transcription and Entity Extraction")

    STUDENT_NAME = "Semih Çelik"
    STUDENT_ID = "150230104"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")
    st.write("""
    Upload a business meeting audio file to:

    1. Transcribe the meeting audio into text.
    2. Extract key entities such as Person, Organizations, Dates, and Locations.
    """)

    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    if uploaded_file is not None:
        st.info("Transcribing the audio file... This may take a minute.")
        text = transcribe_audio(uploaded_file)
        st.subheader("Transcription:")
        st.write(text)

        st.info("Extracting entities...")
        ner_pipeline = load_ner_model()
        extract = extract_entities(text, ner_pipeline)

        st.subheader("Extracted Entities:")
        st.subheader("Organizations (ORGs):")
        st.markdown("- " + "\n- ".join(extract["ORGs"]))
        st.subheader("Locations (LOCs):")
        st.markdown("- " + "\n- ".join(extract["LOCs"]))
        st.subheader("Persons (PERs):")
        st.markdown("- " + "\n- ".join(extract["PERs"]))

if __name__ == "__main__":
    main()