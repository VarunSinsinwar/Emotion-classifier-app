import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, timesteps, 1)
        alpha = K.softmax(e, axis=1)  # attention weights
        context = x * alpha  # apply attention
        context = K.sum(context, axis=1)  # sum over time
        return context

# Load the model
model = tf.keras.models.load_model("emotion_classifier_gru.h5", custom_objects={"Attention": Attention})
emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Feature extraction
def extract_features(file_path, max_pad_len=200):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# UI
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` file to predict the emotion in the speaker's voice.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    try:
        features = extract_features("temp.wav")
        features = features[np.newaxis, ..., np.newaxis]
        prediction = model.predict(features)
        predicted = emotion_classes[np.argmax(prediction)]
        st.success(f"🎯 Predicted Emotion: {predicted}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
