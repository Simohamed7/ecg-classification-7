# app.py
import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, savgol_filter
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
from tensorflow.keras.models import load_model

# Pan-Tompkins (ecgdetectors)
from ecgdetectors import Detectors

# ---------------------------
# Utils: filtres & FrFT
# ---------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smooth_signal(signal, window_length=11, polyorder=3):
    # window_length doit être impair et <= len(signal)
    wl = min(window_length, len(signal) - (1 - len(signal) % 2))
    wl = wl if wl % 2 == 1 else max(3, wl - 1)
    wl = max(3, wl)
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

def frft_magnitude_image(signal_1d, a, target_size=(224, 224)):
    """Trace la magnitude de la FrFT en figure, capture en image PIL 224x224."""
    mag = np.abs(frft(signal_1d, a))

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.plot(mag)
    plt.tight_layout(pad=0)

    # IMPORTANT: utilisation buffer_rgba() puis retrait du canal alpha
    fig.canvas.draw()
    img_array = np.asarray(fig.canvas.buffer_rgba())
    img_array = img_array[:, :, :3]  # RGB
    plt.close(fig)

    img_pil = Image.fromarray(img_array).resize(target_size)
    return img_pil

# ---------------------------
# Chargement modèle (cache)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    return load_model(path)

# ---------------------------
# Chargement ECG
# ---------------------------
def load_signal_from_mat(mat_data):
    """Essaie de trouver une variable ECG plausible dans un .mat."""
    # Priorités classiques: 'val', 'signal', 'ecg', 'data'
    priority_keys = ['val', 'signal', 'ecg', 'data', 'ECG', 'x', 'y']
    for k in priority_keys:
        if k in mat_data:
            arr = np.ravel(np.array(mat_data[k]))
            if arr.size > 0:
                return arr
    # Sinon, première clé non __
    for key in mat_data.keys():
        if not key.startswith("__"):
            arr = np.ravel(np.array(mat_data[key]))
            if arr.size > 0:
                return arr
    raise ValueError("Aucun vecteur signal trouvé dans ce .mat")

# ---------------------------
# Détection R-peaks Pan-Tompkins
# ---------------------------
def detect_rpeaks_pan_tompkins(filtered, fs):
    detectors = Detectors(fs)
    r_peaks = detectors.pan_tompkins_detector(filtered)
    # r_peaks peut contenir des indices hors bornes si signal court → on nettoie:
    r_peaks = np.array([p for p in r_peaks if 0 <= p < len(filtered)], dtype=int)
    return r_peaks

# ---------------------------
# Segmentation de battements
# ---------------------------
def extract_beats(signal_1d, r_peaks, fs, pre_s=0.30, post_s=0.30, max_beats=4):
    """Extrait jusqu'à max_beats fenêtres autour des R-peaks."""
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
# UI
# ---------------------------
st.set_page_config(page_title="ECG: Segmentation & Classification", layout="wide")
st.title("🫀 ECG → Pan-Tompkins → 4 battements → FrFT → Classification")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("Importer (.mat, .csv, .png, .jpg)", type=["mat", "csv", "png", "jpg", "jpeg"])
fs = st.sidebar.number_input("Fréquence d'échantillonnage (Hz)", value=360, min_value=50, max_value=2000, step=10)
lowcut = st.sidebar.number_input("Passe-bande: lowcut (Hz)", value=0.5, step=0.1)
highcut = st.sidebar.number_input("Passe-bande: highcut (Hz)", value=50.0, step=0.5)
use_savgol = st.sidebar.checkbox("Lisser avec Savitzky–Golay", value=True)
sg_window = st.sidebar.slider("SG window", 5, 101, 21, step=2)
sg_poly = st.sidebar.slider("SG polyorder", 2, 7, 3, step=1)
frft_order = st.sidebar.slider("Ordre FrFT (a)", 0.0, 2.0, 1.0, 0.1)
pre_s = st.sidebar.slider("Fenêtre avant R (s)", 0.10, 0.60, 0.30, 0.05)
post_s = st.sidebar.slider("Fenêtre après R (s)", 0.10, 0.60, 0.30, 0.05)
model_path = st.sidebar.text_input("Chemin modèle Keras (.h5)", "best_model_single.h5")
classes_text = st.sidebar.text_input("Noms de classes (séparés par ,)", "")

# Charger le modèle
model = None
model_load_error = None
try:
    model = load_keras_model(model_path)
except Exception as e:
    model_load_error = str(e)

# Espace principal
col_left, col_right = st.columns([1.1, 1])

if uploaded_file is None:
    st.info("➡️ Importez un fichier ECG (.mat/.csv) ou une image (.png/.jpg) pour commencer.")
    if model_load_error:
        st.warning(f"Modèle non chargé: {model_load_error}")
    st.stop()

# Gestion image directe (classification sans segmentation)
if uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
    if model is None:
        st.error("Le modèle n'est pas chargé. Vérifiez le chemin du modèle dans la sidebar.")
        st.stop()

    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_input = np.array(img) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    preds = model.predict(img_input)
    pred_idx = int(np.argmax(preds, axis=1)[0])

    class_names = [c.strip() for c in classes_text.split(",")] if classes_text.strip() else []
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Classe {pred_idx}"

    st.subheader("🖼️ Classification d'image directe")
    st.image(img, caption=f"Image 224×224")
    st.write(f"**Prédiction**: {label}")
    st.write("**Probabilités**:", np.round(preds[0], 3))
    if model_load_error:
        st.caption(f"(Info modèle) {model_load_error}")
    st.stop()

# Charger .mat/.csv → signal 1D
try:
    if uploaded_file.name.endswith(".mat"):
        mat_data = sio.loadmat(uploaded_file)
        signal = load_signal_from_mat(mat_data)
    else:  # .csv
        df = pd.read_csv(uploaded_file)
        # 1ère colonne comme ECG
        signal = df.iloc[:, 0].values.astype(float)
except Exception as e:
    st.error(f"Erreur lors du chargement du signal: {e}")
    st.stop()

signal = np.asarray(signal).astype(float)
if signal.ndim != 1:
    signal = np.ravel(signal)

with col_left:
    st.subheader("Signal brut")
    st.line_chart(signal)

# Filtrage
try:
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    if use_savgol:
        filtered = smooth_signal(filtered, window_length=sg_window, polyorder=sg_poly)
except Exception as e:
    st.error(f"Erreur filtrage: {e}")
    st.stop()

with col_left:
    st.subheader("Signal filtré")
    st.line_chart(filtered)

# Détection R-peaks (Pan-Tompkins)
try:
    r_peaks = detect_rpeaks_pan_tompkins(filtered, fs)
except Exception as e:
    st.error(f"Erreur Pan-Tompkins: {e}")
    st.stop()

if len(r_peaks) < 1:
    st.warning("Aucun R-peak détecté. Ajuste les filtres (lowcut/highcut) ou vérifie fs.")
    st.stop()

# BPM
if len(r_peaks) > 1:
    rr_sec = np.diff(r_peaks) / fs
    bpm = 60.0 / np.mean(rr_sec)
else:
    bpm = 0.0

with col_right:
    st.subheader("💓 Rythme cardiaque")
    st.metric("Fréquence (BPM)", f"{bpm:.1f}")

# Afficher signal + R-peaks
with col_right:
    st.subheader("R-peaks (Pan-Tompkins)")
    fig_peaks, axp = plt.subplots()
    t = np.arange(len(filtered)) / fs
    axp.plot(t, filtered, linewidth=1)
    axp.scatter(r_peaks / fs, filtered[r_peaks], s=20)
    axp.set_xlabel("Temps (s)")
    axp.set_ylabel("Amplitude")
    axp.grid(True, linewidth=0.3)
    st.pyplot(fig_peaks)
    plt.close(fig_peaks)

# Segmentation → 4 battements max
beats, centers = extract_beats(filtered, r_peaks, fs, pre_s=pre_s, post_s=post_s, max_beats=4)
if len(beats) == 0:
    st.warning("Impossible d'extraire des battements. Essayez d'ajuster les fenêtres avant/après R.")
    st.stop()

st.subheader("🧩 Segmentation: 4 battements (max)")
cols = st.columns(len(beats))

# Charger modèle si pas dispo
if model is None:
    st.error("Le modèle n'est pas chargé. Vérifie le chemin dans la sidebar.")
    st.stop()

# Noms de classes (optionnels)
class_names = [c.strip() for c in classes_text.split(",")] if classes_text.strip() else []

# Classification par battement
predicted_indices = []
prob_list = []

for i, beat in enumerate(beats):
    # Eviter signaux vides
    if beat.size < 8:
        with cols[i]:
            st.error(f"Battement {i+1}: fenêtre trop courte.")
        continue

    # Image FrFT 224×224
    img_pil = frft_magnitude_image(beat, frft_order, target_size=(224, 224))

    # Préparer entrée modèle
    img_input = np.array(img_pil) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    preds = model.predict(img_input)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    predicted_indices.append(pred_idx)
    prob_list.append(preds[0])

    # Libellé lisible si fourni
    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else f"Classe {pred_idx}"

    with cols[i]:
        st.image(img_pil, caption=f"Battement {i+1}")
        st.write(f"**Prédiction**: {label}")
        st.write("**Probabilités**:", np.round(preds[0], 3))

# Résumé des classes détectées
if predicted_indices:
    st.markdown("---")
    st.subheader("📊 Résumé (4 battements)")
    # Comptage
    unique, counts = np.unique(predicted_indices, return_counts=True)
    total = len(predicted_indices)
    summary = []
    for u, c in zip(unique, counts):
        name = class_names[u] if class_names and u < len(class_names) else f"Classe {u}"
        summary.append((name, int(c), float(100*c/total)))
    # Affichage tableau
    df_summary = pd.DataFrame(summary, columns=["Classe", "# Battements", "%"])
    st.dataframe(df_summary, hide_index=True)

    # Majority vote
    maj_idx = int(unique[np.argmax(counts)])
    maj_label = class_names[maj_idx] if class_names and maj_idx < len(class_names) else f"Classe {maj_idx}"
    st.success(f"🧠 **Classe majoritaire** (vote sur {total} battements) : {maj_label}")

# Infos modèle si échec de chargement silencieux
if model_load_error:
    st.caption(f"(Info modèle) {model_load_error}")
