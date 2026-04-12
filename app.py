import io
import json

import soundfile as sf
import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="Audio Insights & Entity Extractor", page_icon="🎙️", layout="wide")


# ── Model Loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Whisper Model...")
def load_whisper_model():
    """Load and cache the Whisper ASR pipeline (loaded only once per session)."""
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


@st.cache_resource(show_spinner="Loading NER Model...")
def load_ner_model():
    """Load and cache the BERT-based NER pipeline (loaded only once per session)."""
    return pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )


# ── Core Functions ───────────────────────────────────────────────────────────

def transcribe_audio(uploaded_file) -> str:
    """Read an uploaded audio file as a numpy array and transcribe it via Whisper.

    The file bytes are decoded with *soundfile* so that Whisper receives a proper
    numpy array instead of raw bytes, which would cause a runtime error.
    """
    whisper_pipeline = load_whisper_model()

    # Decode bytes → numpy array (float32) + sample rate
    audio_bytes = uploaded_file.read()
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Whisper expects mono audio; average channels if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Pass the numpy array together with the sampling rate
    result = whisper_pipeline(
        {"array": audio_array, "sampling_rate": sample_rate},
        return_timestamps=True,
    )
    return result["text"]


def extract_entities(text: str, ner_pipeline) -> dict[str, list[str]]:
    """Run NER on *text* and return a dict of entity groups.

    Uses ``truncation=True`` so that texts longer than BERT's 512-token limit
    are handled gracefully instead of raising an error.

    Recognised groups: PER → Persons, ORG → Organizations,
    LOC → Locations, MISC → Miscellaneous.
    """
    ner_results = ner_pipeline(text, truncation=True)

    entities: dict[str, list[str]] = {
        "Persons": [],
        "Organizations": [],
        "Locations": [],
        "Miscellaneous": [],
    }

    label_map = {
        "PER": "Persons",
        "ORG": "Organizations",
        "LOC": "Locations",
        "MISC": "Miscellaneous",
    }

    for result in ner_results:
        group = result["entity_group"]
        word = result["word"]
        target = label_map.get(group)
        if target:
            entities[target].append(word)

    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities


# ── Helpers ──────────────────────────────────────────────────────────────────

def _entities_to_text(entities: dict[str, list[str]]) -> str:
    """Format extracted entities as a human-readable plain-text block."""
    lines: list[str] = []
    for category, items in entities.items():
        lines.append(f"{'─' * 30}")
        lines.append(f"  {category}")
        lines.append(f"{'─' * 30}")
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append("  (none)")
        lines.append("")
    return "\n".join(lines)


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("🎙️ Audio Insights & Entity Extractor")
    st.markdown(
        """
    Upload a business meeting audio file to:
    1. **Transcribe** the meeting audio into text automatically.
    2. **Extract** key entities such as Persons, Organizations, Locations, and more from the transcription.
    """
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

    if uploaded_file is not None:
        # ── Audio preview ────────────────────────────────────────────────
        st.audio(uploaded_file)

        # ── Transcription ────────────────────────────────────────────────
        try:
            with st.spinner("🎙️ Transcribing the audio file... This may take a moment."):
                text = transcribe_audio(uploaded_file)
        except Exception as exc:
            st.error(f"❌ Transcription failed: {exc}")
            return

        if not text or text.isspace():
            st.warning("⚠️ The transcription returned empty text. The audio may be silent or too short.")
            return

        st.success("Transcription Complete!")

        with st.expander("📝 View Full Transcription", expanded=True):
            st.write(text)

        st.download_button(
            label="💾 Download Transcription (.txt)",
            data=text,
            file_name="transcription.txt",
            mime="text/plain",
        )

        # ── NER ──────────────────────────────────────────────────────────
        try:
            with st.spinner("🔍 Extracting Named Entities..."):
                ner_model = load_ner_model()
                extracted_entities = extract_entities(text, ner_model)
        except Exception as exc:
            st.error(f"❌ Entity extraction failed: {exc}")
            return

        st.success("Entity Extraction Complete!")

        # ── Metrics ──────────────────────────────────────────────────────
        st.subheader("Extracted Entities")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👤 Persons", len(extracted_entities["Persons"]))
        m2.metric("🏢 Organizations", len(extracted_entities["Organizations"]))
        m3.metric("📍 Locations", len(extracted_entities["Locations"]))
        m4.metric("🏷️ Miscellaneous", len(extracted_entities["Miscellaneous"]))

        # ── Entity columns ───────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### 👤 Persons")
            if extracted_entities["Persons"]:
                for per in extracted_entities["Persons"]:
                    st.markdown(f"- {per}")
            else:
                st.write("*No persons found.*")

        with col2:
            st.markdown("### 🏢 Organizations")
            if extracted_entities["Organizations"]:
                for org in extracted_entities["Organizations"]:
                    st.markdown(f"- {org}")
            else:
                st.write("*No organizations found.*")

        with col3:
            st.markdown("### 📍 Locations")
            if extracted_entities["Locations"]:
                for loc in extracted_entities["Locations"]:
                    st.markdown(f"- {loc}")
            else:
                st.write("*No locations found.*")

        with col4:
            st.markdown("### 🏷️ Miscellaneous")
            if extracted_entities["Miscellaneous"]:
                for misc in extracted_entities["Miscellaneous"]:
                    st.markdown(f"- {misc}")
            else:
                st.write("*No miscellaneous entities found.*")

        # ── Download entities ────────────────────────────────────────────
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="💾 Download Entities (.txt)",
                data=_entities_to_text(extracted_entities),
                file_name="entities.txt",
                mime="text/plain",
            )
        with dl2:
            st.download_button(
                label="💾 Download Entities (.json)",
                data=json.dumps(extracted_entities, indent=2, ensure_ascii=False),
                file_name="entities.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
