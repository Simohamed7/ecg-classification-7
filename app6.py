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
    filtered = bandpass_filter(signal, 5, 15, fs)
    diff = np.diff(filtered)
    diff = np.append(diff, 0)
    squared = diff ** 2
    window = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
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
# Streamlit interface
# ---------------------------
st.set_page_config(page_title="ECG Segmentation & Classification", layout="wide")
st.title("🫀 ECG → Pan-Tompkins → 4 battements → FrFT → Classification")

# Sidebar paramètres
st.sidebar.header("⚙️ Paramètres")
uploaded_signal = st.sidebar.file_uploader("Importer ECG (.mat, .csv, .png, .jpg)", type=["mat","csv","png","jpg","jpeg"])
model_file = st.sidebar.file_uploader("Charger modèle Keras (.h5)", type=["h5"])

fs = st.sidebar.number_input("Fréquence d'échantillonnage (Hz)", value=360)
use_savgol = st.sidebar.checkbox("Lissage Savitzky–Golay", value=True)
sg_window = st.sidebar.slider("SG window", 5, 101, 21, step=2)
sg_poly = st.sidebar.slider("SG polyorder", 2, 7, 3)
pre_s = st.sidebar.slider("Fenêtre avant R (s)", 0.10,0.60,0.30,0.05)
post_s = st.sidebar.slider("Fenêtre après R (s)",0.10,0.60,0.30,0.05)

# Classes
class_names = ["F3", "N0", "Q4", "S1", "V2"]
class_full_names = {"N0":"NORMAL","S1":"SUPRAVENTRICULAR","V2":"VENTRICULAR","F3":"FUSION","Q4":"UNKNOWN"}

# Vérifier upload modèle
if model_file is None:
    st.warning("Veuillez uploader votre modèle `.h5` pour continuer.")
    st.stop()
else:
    try:
        model = load_model(model_file)
        st.success("Modèle chargé avec succès !")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        st.stop()

# Vérifier upload signal
if uploaded_signal is None:
    st.info("Chargez un fichier ECG pour commencer.")
    st.stop()

# Charger signal
if uploaded_signal.name.lower().endswith(".mat"):
    mat_data = sio.loadmat(uploaded_signal)
    for key in mat_data:
        if not key.startswith("__"):
            signal = np.ravel(mat_data[key])
            break
elif uploaded_signal.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded_signal)
    signal = df.iloc[:,0].values.astype(float)
elif uploaded_signal.name.lower().endswith((".png","jpg","jpeg")):
    st.warning("Pour images directes, classification sans segmentation.")
    img = Image.open(uploaded_signal).convert("RGB").resize((224,224))
    img_input = np.expand_dims(np.array(img)/255.0,0)
    preds = model.predict(img_input)
    pred_idx = np.argmax(preds,axis=1)[0]
    label_full = class_full_names.get(class_names[pred_idx], f"Classe {pred_idx}")
    st.image(img, caption="Image 224×224")
    st.write("Classe:", label_full)
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

# Alpha list 0→1 pas 0.01
alpha_list = np.arange(0,1.01,0.01)

# Segmentation et classification
beats, centers = extract_beats(filtered, r_peaks, fs, pre_s, post_s, max_beats=4)
st.subheader("Segmentation 4 battements max")

for i, beat in enumerate(beats):
    st.markdown(f"### Battement {i+1}")
    beat_results = []
    for a in alpha_list:
        img_pil = frft_magnitude_image(beat, a, (224,224))
        img_input = np.expand_dims(np.array(img_pil)/255.0,0)
        preds = model.predict(img_input)
        pred_idx = np.argmax(preds,axis=1)[0]
        beat_results.append((a, pred_idx, preds[0]))

    # Afficher image et prédiction pour alpha=0.5
    alpha_example = 0.5
    idx_example = int(alpha_example/0.01)
    img_example = frft_magnitude_image(beat, alpha_example, (224,224))
    st.image(img_example, caption=f"Alpha={alpha_example}")
    pred_idx = beat_results[idx_example][1]
    label_full = class_full_names.get(class_names[pred_idx], f"Classe {pred_idx}")
    st.write("Classe:", label_full)
    st.write("Probabilités:", np.round(beat_results[idx_example][2],3))

    # Tableau alpha vs classe vs probabilité
    df_alpha = pd.DataFrame({
        "Alpha": [r[0] for r in beat_results],
        "Classe": [class_names[r[1]] for r in beat_results],
        "Probabilité_max": [np.max(r[2]) for r in beat_results]
    })
    st.dataframe(df_alpha)

# Résumé global
st.subheader("Résumé des 4 battements")
predicted_indices = [np.argmax(model.predict(np.expand_dims(np.array(frft_magnitude_image(beat, 0.5, (224,224)))/255.0,0))) for beat in beats]
unique, counts = np.unique(predicted_indices, return_counts=True)
df_summary = pd.DataFrame({
    "Classe": [class_full_names.get(class_names[u], f"Classe {u}") for u in unique],
    "# Battements": counts,
    "%": [100*c/len(predicted_indices) for c in counts]
})
st.dataframe(df_summary)
