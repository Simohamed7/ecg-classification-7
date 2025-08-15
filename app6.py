import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------
# Filtrage et FrFT
# ---------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smooth_signal(signal, window_length=11, polyorder=3):
    wl = min(window_length, len(signal) - (1 - len(signal) % 2))
    wl = wl if wl % 2 == 1 else max(3, wl - 1)
    poly = min(polyorder, max(2, wl - 1))
    return savgol_filter(signal, wl, poly)

def frft(f, a):
    N = len(f)
    shft = np.arange(N)
    shft = np.where(shft > N/2, shft - N, shft)
    alpha = a * pi / 2
    if a == 0: return f
    if a == 1: return np.fft.fft(f)
    if a == 2: return np.flipud(f)
    if a == -1: return np.fft.ifft(f)
    tana2 = np.tan(alpha/2)
    sina = np.sin(alpha)
    chirp1 = np.exp(-1j * pi * (shft**2) * tana2 / N)
    f2 = f * chirp1
    F = np.fft.fft(f2 * np.exp(-1j * pi * (shft**2) / (N * sina)))
    F = F * np.exp(-1j * pi * (shft**2) * tana2 / N)
    return F

def frft_magnitude_image(signal_1d, a, target_size=(224,224)):
    mag = np.abs(frft(signal_1d, a))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.plot(mag)
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    img_array = np.asarray(fig.canvas.buffer_rgba())
    img_array = img_array[:, :, :3]  # RGB seulement
    plt.close(fig)
    img_pil = Image.fromarray(img_array).resize(target_size)
    return img_pil

# ---------------------------
# Pan-Tompkins simplifié
# ---------------------------
def pan_tompkins_detect(signal, fs):
    # 1. Filtrage bande-pass
    filtered = bandpass_filter(signal, 5, 15, fs)
    # 2. Dérivée
    diff = np.diff(filtered)
    diff = np.append(diff, 0)
    # 3. Carré
    squared = diff ** 2
    # 4. Intégration sur fenêtre (150 ms)
    window = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
    # 5. Détection pics
    peaks, _ = find_peaks(integrated, distance=0.3*fs, height=np.mean(integrated))
    return peaks

def extract_beats(signal_1d, r_peaks, fs, pre_s=0.3, post_s=0.3, max_beats=4):
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    beats = []
    centers = []
    count = min(max_beats, len(r_peaks))
    for i in range(count):
        c = r_peaks[i]
        start = max(0, c - pre)
        end = min(len(signal_1d), c + post)
        beats.append(signal_1d[start:end])
        centers.append(c)
    return beats, centers

# ---------------------------
# Charger modèle Keras
# ---------------------------
@st.cache_resource
def load_keras_model(path):
    return load_model(path)

# ---------------------------
# Interface Streamlit
# ---------------------------
st.set_page_config(page_title="ECG: Segmentation & Classification", layout="wide")
st.title("🫀 ECG → Pan-Tompkins → 4 battements → FrFT → Classification")

st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("Importer (.mat, .csv, .png, .jpg)", type=["mat","csv","png","jpg","jpeg"])
fs = st.sidebar.number_input("Fréquence d'échantillonnage (Hz)", value=360)
use_savgol = st.sidebar.checkbox("Lissage Savitzky–Golay", value=True)
sg_window = st.sidebar.slider("SG window", 5, 101, 21, step=2)
sg_poly = st.sidebar.slider("SG polyorder", 2, 7, 3)
frft_order = st.sidebar.slider("Ordre FrFT (a)", 0.0, 2.0, 1.0, 0.1)
pre_s = st.sidebar.slider("Fenêtre avant R (s)", 0.10,0.60,0.30,0.05)
post_s = st.sidebar.slider("Fenêtre après R (s)",0.10,0.60,0.30,0.05)
model_path = st.sidebar.text_input("Chemin modèle Keras (.h5)", "best_model_single.h5")
classes_text = st.sidebar.text_input("Noms des classes (séparés par ,)", "")

# Charger modèle
try:
    model = load_keras_model(model_path)
except Exception as e:
    st.error(f"Erreur chargement modèle: {e}")
    st.stop()

# Charger signal
if uploaded_file is None:
    st.info("Chargez un fichier ECG (.mat/.csv) ou image (.png/.jpg).")
    st.stop()

if uploaded_file.name.lower().endswith(".mat"):
    mat_data = sio.loadmat(uploaded_file)
    for key in mat_data:
        if not key.startswith("__"):
            signal = np.ravel(mat_data[key])
            break
elif uploaded_file.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded_file)
    signal = df.iloc[:,0].values.astype(float)
elif uploaded_file.name.lower().endswith((".png","jpg","jpeg")):
    st.warning("Pour images directes, classification sans segmentation.")
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    img_input = np.expand_dims(np.array(img)/255.0,0)
    preds = model.predict(img_input)
    pred_idx = np.argmax(preds,axis=1)[0]
    class_names = [c.strip() for c in classes_text.split(",")] if classes_text.strip() else []
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Classe {pred_idx}"
    st.image(img, caption="Image 224×224")
    st.write("Classe:", label)
    st.write("Probabilités:", np.round(preds[0],3))
    st.stop()

signal = np.asarray(signal).astype(float)
if signal.ndim != 1:
    signal = np.ravel(signal)

# Signal brut
st.subheader("Signal brut")
st.line_chart(signal)

# Filtrage et lissage
filtered = bandpass_filter(signal, 0.5, 50, fs)
if use_savgol:
    filtered = smooth_signal(filtered, sg_window, sg_poly)
st.subheader("Signal filtré")
st.line_chart(filtered)

# Détection R-peaks
r_peaks = pan_tompkins_detect(filtered, fs)
if len(r_peaks) < 1:
    st.warning("Aucun R-peak détecté.")
    st.stop()

# Calcul BPM
if len(r_peaks) > 1:
    rr_sec = np.diff(r_peaks)/fs
    bpm = 60.0 / np.mean(rr_sec)
else:
    bpm = 0.0
st.metric("💓 Rythme cardiaque (BPM)", f"{bpm:.1f}")

# Segmentation et classification
beats, centers = extract_beats(filtered, r_peaks, fs, pre_s, post_s, max_beats=4)
st.subheader("Segmentation 4 battements max")
cols = st.columns(len(beats))

class_names = [c.strip() for c in classes_text.split(",")] if classes_text.strip() else []
predicted_indices = []

for i, beat in enumerate(beats):
    img_pil = frft_magnitude_image(beat, frft_order, (224,224))
    img_input = np.expand_dims(np.array(img_pil)/255.0,0)
    preds = model.predict(img_input)
    pred_idx = np.argmax(preds,axis=1)[0]
    predicted_indices.append(pred_idx)
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Classe {pred_idx}"
    with cols[i]:
        st.image(img_pil, caption=f"Battement {i+1}")
        st.write("Classe:", label)
        st.write("Probabilités:", np.round(preds[0],3))

# Résumé
if predicted_indices:
    st.subheader("Résumé des 4 battements")
    unique, counts = np.unique(predicted_indices, return_counts=True)
    df_summary = pd.DataFrame({
        "Classe": [class_names[u] if class_names and u<len(class_names) else f"Classe {u}" for u in unique],
        "# Battements": counts,
        "%": [100*c/len(predicted_indices) for c in counts]
    })
    st.dataframe(df_summary)
