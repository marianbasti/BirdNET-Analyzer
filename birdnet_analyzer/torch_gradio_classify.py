import os
import torch
import torchaudio
import numpy as np
from birdnet_analyzer.torch_model import BirdNetTorchModel

def classify_interface(audio_dir, model_path, window_size_sec, hop_size_sec, threshold=0.5):
    # Scan for audio files
    if not os.path.isdir(audio_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))]
    if not audio_files:
        return {"error": "No se encontraron archivos de audio en el directorio proporcionado."}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model and infer class names from checkpoint (if possible)
    try:
        # Try to infer number of classes from checkpoint shape
        state = torch.load(model_path, map_location=device)
        # Try to get num_classes from classifier weight shape
        num_classes = state['classifier.weight'].shape[0] if 'classifier.weight' in state else 10
        model = BirdNetTorchModel(num_classes=num_classes)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}
    # Class names: try to get from subdirs, else use generic
    class_names = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]
    # Sliding window params
    sample_rate = 48000
    window_size = int(float(window_size_sec) * sample_rate)
    hop_size = int(float(hop_size_sec) * sample_rate)
    results = {}
    for fname in audio_files:
        path = os.path.join(audio_dir, fname)
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        if waveform.ndim > 1:
            waveform = waveform[0]
        total_len = waveform.shape[-1]
        preds_per_window = []
        times = []
        for start in range(0, total_len - window_size + 1, hop_size):
            window = waveform[..., start:start+window_size]
            if window.shape[-1] < window_size:
                pad_width = window_size - window.shape[-1]
                window = torch.nn.functional.pad(window, (0, pad_width))
            window = window.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(window)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds_per_window.append(probs)
            times.append((start/sample_rate, (start+window_size)/sample_rate))
        if not preds_per_window:
            results[fname] = {}
            continue
        preds_per_window = np.stack(preds_per_window)
        # For each class, find windows where prob > threshold
        detected = {}
        for i, cname in enumerate(class_names):
            intervals = []
            for j, prob in enumerate(preds_per_window[:, i]):
                if prob > threshold:
                    intervals.append(times[j])
            if intervals:
                detected[cname] = intervals
        results[fname] = detected
    return results

# Utility function to list classes in a directory
def list_classes(data_dir):
    import os
    if not os.path.isdir(data_dir):
        return ""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return ""
    return ", ".join(class_names)