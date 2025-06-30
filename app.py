import gradio as gr
from pydub import AudioSegment
import numpy as np
from scipy.signal import fftconvolve
import tempfile

def bandpass_filter(audio, lowcut=300, highcut=3400):
    audio = audio.high_pass_filter(lowcut)
    audio = audio.low_pass_filter(highcut)
    return audio

def bitcrush(audio, bit_depth=8):
    arr = np.array(audio.get_array_of_samples())
    factor = 2 ** (16 - bit_depth)
    arr = ((arr // factor) * factor).astype(np.int16)
    return audio._spawn(arr.tobytes())

def convolve_ir(audio, ir_bytes):
    if ir_bytes is None:
        return audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ir_tempf:
        ir_tempf.write(ir_bytes.read())
        ir_tempf.flush()
        ir_audio = AudioSegment.from_wav(ir_tempf.name).set_channels(audio.channels).set_frame_rate(audio.frame_rate)
        dry = np.array(audio.get_array_of_samples(), dtype=np.float32)
        ir = np.array(ir_audio.get_array_of_samples(), dtype=np.float32)
        wet = fftconvolve(dry, ir[:min(len(ir), 2000)], mode="full")
        wet = wet / np.max(np.abs(wet)) * 32767
        wet = wet.astype(np.int16)
        return audio._spawn(wet[:len(dry)].tobytes())

def lfo_sample_hold_dropout(audio, interval_ms=100, dropout_prob=0.25, drop_gain_db=-60):
    segments = []
    pos = 0
    while pos < len(audio):
        segment = audio[pos:pos+interval_ms]
        if np.random.rand() < dropout_prob:
            segment = segment + drop_gain_db
        segments.append(segment)
        pos += interval_ms
    return sum(segments)

def process_audio(input_wav, ir_file, interval_ms, dropout_prob, bit_depth):
    # Save input for pydub
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        temp_input.write(input_wav.read())
        temp_input.flush()
        audio = AudioSegment.from_wav(temp_input.name).set_channels(1).set_frame_rate(16000)
    audio = bandpass_filter(audio)
    audio = bitcrush(audio, bit_depth=bit_depth)
    audio = convolve_ir(audio, ir_file)
    audio = lfo_sample_hold_dropout(audio, interval_ms=interval_ms, dropout_prob=dropout_prob)
    # Export output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        audio.export(temp_output.name, format="wav")
        return temp_output.name

title = "Spotty Service Pro Demo (Gradio)"
desc = "Upload a WAV file to hear ultra-realistic 'bad phone' FX—including phone EQ, bitcrush, convolution, and signal dropouts!<br>Optional: Upload a Phone IR WAV file for even more realism."

iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.File(label="Input WAV", file_types=[".wav"]),
        gr.File(label="Phone IR WAV (optional)", file_types=[".wav"], optional=True),
        gr.Slider(50, 400, value=120, step=10, label="Dropout Interval (ms)"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Dropout Probability"),
        gr.Slider(4, 16, value=8, step=1, label="Bit Depth"),
    ],
    outputs=gr.Audio(type="filepath", label="Processed Audio"),
    title=title,
    description=desc,
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch()
