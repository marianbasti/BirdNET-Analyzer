import gradio as gr
import os
import numpy as np
import torch
import torchaudio
from birdnet_analyzer.torch_model import BirdNetTorchModel

def classify_interface(audio_dir, model_path, window_size_sec, hop_size_sec, threshold=0.5):
    """Classification interface for sliding window detection."""
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

def create_classify_tab():
    """Create the classification tab interface."""
    with gr.TabItem("Clasificar"):
        gr.Markdown("## Clasificación de Audio (Ventana Deslizante)")
        gr.Markdown("Ingrese la ruta a un directorio que contenga archivos de audio (grabaciones largas, sin recortar). Seleccione un punto de control de modelo entrenado. Configure el tamaño de ventana y salto en segundos. El sistema ejecutará una ventana deslizante e informará las clases detectadas y sus intervalos de tiempo.")
        
        with gr.Row():
            classify_dir_input = gr.Textbox(
                label="Ruta del Directorio de Audio",
                placeholder="/ruta/a/directorio_audio",
                info="Directorio que contiene archivos de audio a clasificar"
            )
            classify_model_input = gr.File(
                label="Punto de Control del Modelo (.pt)",
            )
        
        with gr.Row():
            classify_window_input = gr.Number(
                label="Tamaño de Ventana (seg)",
                value=5,
                info="Duración de cada ventana de análisis en segundos"
            )
            classify_hop_input = gr.Number(
                label="Tamaño de Salto (seg)",
                value=2.5,
                info="Cantidad de segundos que se avanza entre ventanas"
            )
            classify_thresh_input = gr.Number(
                label="Umbral de Detección",
                value=0.5,
                info="Probabilidad mínima para considerar una clase como detectada"
            )
        
        classify_output = gr.JSON(label="Clases Detectadas e Intervalos")
        classify_btn = gr.Button("Clasificar Audios")
        classify_btn.click(classify_interface, inputs=[classify_dir_input, classify_model_input, classify_window_input, classify_hop_input, classify_thresh_input], outputs=classify_output)
