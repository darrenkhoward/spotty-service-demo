import streamlit as st
from pydub import AudioSegment
import numpy as np
import tempfile

def lfo_sample_hold_dropout(audio, interval_ms=100, dropout_prob=0.25, drop_gain_db=-60):
    segments = []
    pos = 0
    while pos < len(audio):
        segment = audio[pos:pos+interval_ms]
        if np.random.rand() < dropout_prob:
            segment = segment + drop_gain_db  # mute/dropout
        segments.append(segment)
        pos += interval_ms
    return sum(segments)

st.title("Spotty Service Audio FX Demo")
st.write("Upload a WAV file. It will add 'spotty service' cell dropouts.")

uploaded_file = st.file_uploader("Upload a WAV file", type="wav")
interval_ms = st.slider("Dropout Interval (ms)", 50, 400, 120, 10)
dropout_prob = st.slider("Dropout Probability", 0.0, 1.0, 0.3, 0.01)

if uploaded_file is not None:
    audio = AudioSegment.from_wav(uploaded_file)
    out_audio = lfo_sample_hold_dropout(audio, interval_ms=interval_ms, dropout_prob=dropout_prob)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tempf:
        out_audio.export(tempf.name, format="wav")
        st.audio(tempf.name)
        st.download_button(label="Download Processed Audio", data=open(tempf.name, "rb"), file_name="spotty_service.wav", mime="audio/wav")
