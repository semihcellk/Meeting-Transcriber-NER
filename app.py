import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Meeting Insights Extractor", page_icon="🎙️", layout="wide")

# Load the Whisper Model with caching
@st.cache_resource(show_spinner="Loading Whisper Model...")
def load_whisper_model():
    pipe1 = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return pipe1

# Load the NER Model with caching
@st.cache_resource(show_spinner="Loading NER Model...")
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
    entities = {"Organizations": [], "Locations": [], "Persons": []}

    for result in ner_model:
        entity_group = result["entity_group"]
        entity_word = result["word"]

        if entity_group == "ORG":
            entities["Organizations"].append(entity_word)
        elif entity_group == "LOC":
            entities["Locations"].append(entity_word)
        elif entity_group == "PER":
            entities["Persons"].append(entity_word)

    # Remove duplicates
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
        
    return entities

def main():
    st.title("🎙️ Audio Insights & Entity Extractor")
    st.markdown("""
    Upload a business meeting audio file to:
    1. **Transcribe** the meeting audio into text automatically.
    2. **Extract** key entities such as Persons, Organizations, and Locations from the transcription.
    """)

    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    
    if uploaded_file is not None:
        with st.spinner("🎙️ Transcribing the audio file... This may take a moment."):
            text = transcribe_audio(uploaded_file)
            
        st.success("Transcription Complete!")
        
        # Display Transcription
        with st.expander("📝 View Full Transcription", expanded=True):
            st.write(text)

        with st.spinner("🔍 Extracting Named Entities..."):
            ner_pipeline = load_ner_model()
            extracted_entities = extract_entities(text, ner_pipeline)
            
        st.success("Entity Extraction Complete!")

        # Display Entities using columns
        st.subheader("Extracted Entities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏢 Organizations")
            if extracted_entities["Organizations"]:
                for org in extracted_entities["Organizations"]:
                    st.markdown(f"- {org}")
            else:
                st.write("*No organizations found.*")
                
        with col2:
            st.markdown("### 📍 Locations")
            if extracted_entities["Locations"]:
                for loc in extracted_entities["Locations"]:
                    st.markdown(f"- {loc}")
            else:
                st.write("*No locations found.*")

        with col3:
            st.markdown("### 👤 Persons")
            if extracted_entities["Persons"]:
                for per in extracted_entities["Persons"]:
                    st.markdown(f"- {per}")
            else:
                st.write("*No persons found.*")

if __name__ == "__main__":
    main()
