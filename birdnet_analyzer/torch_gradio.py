from sklearn.manifold import TSNE
def tsne_visualization_interface(data_dir, model_path, n_samples=500):
    import torch
    import torchaudio
    from birdnet_analyzer.torch_model import EfficientNetBackbone, BirdNETMelSpecLayer
    import os
    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model backbone
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        backbone = EfficientNetBackbone(1, 1024).to(device)
        state = torch.load(model_path, map_location=device)
        backbone.load_state_dict(state)
        backbone.eval()
    except Exception as e:
        return {"error": f"Error al cargar el backbone: {e}"}
    # Collect audio files and labels
    audio_paths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                audio_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    if not audio_paths:
        return {"error": "No se encontraron archivos de audio en los subdirectorios de clase."}
    # Subsample for visualization
    if len(audio_paths) > n_samples:
        import numpy as np
        idxs = np.random.choice(len(audio_paths), n_samples, replace=False)
        audio_paths = [audio_paths[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    # Extract features
    features = []
    y_labels = []
    spec_layer = BirdNETMelSpecLayer().to(device)
    for path, label in zip(audio_paths, labels):
        try:
            waveform, sr = torchaudio.load(path)
            if sr != 48000:
                waveform = torchaudio.functional.resample(waveform, sr, 48000)
            if waveform.ndim > 1:
                waveform = waveform[0]
            waveform = waveform.unsqueeze(0).to(device)
            with torch.no_grad():
                spec = spec_layer(waveform)
                emb = backbone(spec)
                features.append(emb.cpu().numpy()[0])
                y_labels.append(label)
        except Exception:
            continue
    if not features:
        return {"error": "No se extrajeron características."}
    import numpy as np
    features = np.stack(features)
    y_labels = np.array(y_labels)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    embedding = tsne.fit_transform(features)
    # Plot
    import matplotlib.pyplot as plt
    import io
    import PIL.Image
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, cname in enumerate(class_names):
        mask = y_labels == idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=cname, alpha=0.7, s=20)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Proyección t-SNE de las características de la red')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img
def classify_interface(audio_dir, model_path, window_size_sec, hop_size_sec, threshold=0.5):
    import os
    import torch
    import torchaudio
    from birdnet_analyzer.torch_model import BirdNetTorchModel
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

import numpy as np
import umap
import matplotlib.pyplot as plt
import io
import gradio as gr
import torch
import torchaudio
from birdnet_analyzer.torch_model import BirdNetTorchModel
from birdnet_analyzer.torch_pretrain_utils import SimCLRPretrainer, UnlabeledAudioDataset, collate_fn

# Add request: gr.Request to function signature
def pretrain_interface(data_dir, epochs, batch_size, learning_rate, save_every_epochs=0, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None): # Added request
    import os
    from torch.utils.data import DataLoader



    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    try:
        dataset = UnlabeledAudioDataset(data_dir)
        if len(dataset) == 0:
            return {"error": "No se encontraron archivos de audio en el directorio proporcionado."}
        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, collate_fn=collate_fn)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrainer = SimCLRPretrainer(device=device)
        # Handle output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'pretrained_backbone.pt')
            checkpoint_prefix = os.path.join(output_dir, 'checkpoint_pretrain_epoch')
        else:
            save_path = 'pretrained_backbone.pt'
            checkpoint_prefix = 'checkpoint_pretrain_epoch'
        # Try to call with checkpoint_prefix if supported
        try:
            pretrainer.train(dataloader, epochs=int(epochs), lr=float(learning_rate), save_path=save_path, checkpoint_every=int(save_every_epochs), checkpoint_prefix=checkpoint_prefix)
        except TypeError:
            pretrainer.train(dataloader, epochs=int(epochs), lr=float(learning_rate), save_path=save_path, checkpoint_every=int(save_every_epochs))
    
    except Exception as e:
        return {"error": f"Error en el preentrenamiento: {e}"}

# Pad or truncate waveform to fixed length (1 minute = 2,880,000 samples at 48kHz)
def pad_or_truncate(waveform, target_length=2880000):
    length = waveform.shape[-1]
    if length == target_length:
        return waveform
    elif length > target_length:
        return waveform[..., :target_length]
    else:
        pad_width = target_length - length
        return torch.nn.functional.pad(waveform, (0, pad_width))


# Add request: gr.Request to function signature
def train_interface(data_dir, model_path, epochs, batch_size, learning_rate, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None): # Added request
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from birdnet_analyzer.torch_train_utils import AudioDataset, train_model
    import torchaudio
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model
    try:
        model = BirdNetTorchModel(num_classes=len(class_names))
        if model_path is not None:
            try:
                state = torch.load(model_path, map_location=device)
                # Try to load as full model first
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    # If failed, try loading as backbone only
                    if hasattr(model, 'backbone'):
                        model.backbone.load_state_dict(state, strict=False)
                        print("Se cargaron los pesos del backbone para el fine-tuning.")
                    else:
                        raise e
            except Exception as e:
                return {"error": f"Error al cargar el modelo: {e}"}
        model = model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}
    # Collect all audio files and their labels
    audio_paths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                audio_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    if not audio_paths:
        return {"error": "No se encontraron archivos de audio en los subdirectorios de clase."}
    # Load all audio data and labels
    audio_tensors = []
    label_tensors = []
    for path, label_idx in zip(audio_paths, labels):
        waveform, sr = torchaudio.load(path)
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
        if waveform.ndim > 1:
            waveform = waveform[0]  # mono
        waveform = pad_or_truncate(waveform, 2880000)
        audio_tensors.append(waveform)
        # One-hot encoding for multi-class
        label_vec = torch.zeros(len(class_names))
        label_vec[label_idx] = 1.0
        label_tensors.append(label_vec)
    audio_data = torch.stack(audio_tensors)
    label_data = torch.stack(label_tensors)
    # Stratified split train/val
    import numpy as np
    from sklearn.model_selection import train_test_split
    idxs = np.arange(len(audio_data))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=42)
    train_audio, val_audio = audio_data[train_idx], audio_data[val_idx]
    train_labels, val_labels = label_data[train_idx], label_data[val_idx]
    train_ds = AudioDataset(train_audio, train_labels)
    val_ds = AudioDataset(val_audio, val_labels)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size))
    # Train model
    from birdnet_analyzer.torch_train_utils import evaluate_model
    # Handle output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        checkpoint_prefix = os.path.join(output_dir, 'checkpoint_finetune_epoch')
    else:
        best_model_path = 'best_model.pt'
        checkpoint_prefix = 'checkpoint_finetune_epoch'
    try:
        # NOTE: train_model would need to be modified to accept a stop_callback
        # e.g., stop_callback=lambda: request.is_cancelled if request else False
        train_model(model, train_loader, val_loader, epochs=int(epochs), lr=float(learning_rate), device=device, progress=progress, best_model_path=best_model_path, checkpoint_prefix=checkpoint_prefix)
    except TypeError: # Fallback for older train_model without checkpointing args
        train_model(model, train_loader, val_loader, epochs=int(epochs), lr=float(learning_rate), device=device, progress=progress)



    # After training, evaluate on val set
    val_loss, val_metrics = evaluate_model(model, val_loader, device, return_metrics=True)
    results = {"val_loss": val_loss, "val_metrics": val_metrics, "class_names": class_names, "best_model_path": best_model_path}
    return results

def eval_interface(data_dir, model_path):
    import os
    import numpy as np
    import torch
    import torchaudio
    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = BirdNetTorchModel(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}
    # Collect all audio files and their labels
    audio_paths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                audio_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    if not audio_paths:
        return {"error": "No se encontraron archivos de audio en los subdirectorios de clase."}
    # Load all audio data and labels
    audio_tensors = []
    label_tensors = []
    for path, label_idx in zip(audio_paths, labels):
        waveform, sr = torchaudio.load(path)
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
        if waveform.ndim > 1:
            waveform = waveform[0]  # mono
        waveform = pad_or_truncate(waveform, 2880000)
        audio_tensors.append(waveform)
        # One-hot encoding for multi-class
        label_vec = torch.zeros(len(class_names))
        label_vec[label_idx] = 1.0
        label_tensors.append(label_vec)
    audio_data = torch.stack(audio_tensors)
    label_data = torch.stack(label_tensors)
    # Evaluate model with metrics
    from torch.utils.data import DataLoader, Dataset
    from birdnet_analyzer.torch_train_utils import AudioDataset, evaluate_model
    val_ds = AudioDataset(audio_data, label_data)
    val_loader = DataLoader(val_ds, batch_size=8)
    val_loss, val_metrics = evaluate_model(model, val_loader, device, return_metrics=True)
    results = {"val_loss": val_loss, "val_metrics": val_metrics, "class_names": class_names}
    return results

# --- UMAP Visualization Tab ---
def umap_visualization_interface(data_dir, model_path, n_samples=500):
    import torch
    import torchaudio
    from birdnet_analyzer.torch_model import EfficientNetBackbone, BirdNETMelSpecLayer
    import os
    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model backbone
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        backbone = EfficientNetBackbone(1, 1024).to(device)
        state = torch.load(model_path, map_location=device)
        backbone.load_state_dict(state)
        backbone.eval()
    except Exception as e:
        return {"error": f"Error al cargar el backbone: {e}"}
    # Collect audio paths and labels
    audio_paths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                audio_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    if not audio_paths:
        return {"error": "No se encontraron archivos de audio en los subdirectorios de clase."}
    # Subsample for visualization
    if len(audio_paths) > n_samples:
        idxs = np.random.choice(len(audio_paths), n_samples, replace=False)
        audio_paths = [audio_paths[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    # Extract features
    features = []
    y_labels = []
    spec_layer = BirdNETMelSpecLayer().to(device)
    for path, label in zip(audio_paths, labels):
        try:
            waveform, sr = torchaudio.load(path)
            if sr != 48000:
                waveform = torchaudio.functional.resample(waveform, sr, 48000)
            if waveform.ndim > 1:
                waveform = waveform[0]
            waveform = waveform.unsqueeze(0).to(device)
            with torch.no_grad():
                spec = spec_layer(waveform)
                emb = backbone(spec)
                features.append(emb.cpu().numpy()[0])
                y_labels.append(label)
        except Exception:
            continue
    if not features:
        return {"error": "No se extrajeron características."}
    features = np.stack(features)
    y_labels = np.array(y_labels)
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, cname in enumerate(class_names):
        mask = y_labels == idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=cname, alpha=0.7, s=20)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Proyección UMAP de las características de la red')
    plt.tight_layout()
    import PIL.Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador de Audio BirdNet (PyTorch)")
    with gr.Tabs():
        with gr.TabItem("Preentrenar"):
            gr.Markdown("## Preentrenamiento Auto-supervisado (SimCLR)")
            gr.Markdown("Ingrese la ruta a un directorio que contenga archivos de audio no etiquetados (cualquier estructura de carpetas). Configure los parámetros de preentrenamiento. El backbone se guardará como 'pretrained_backbone.pt'.")
            with gr.Row():
                pretrain_dir_input = gr.Textbox(
                    label="Ruta del Directorio de Datos No Etiquetados",
                    placeholder="/ruta/a/datos_no_etiquetados",
                    info="Directorio que contiene archivos de audio no etiquetados para preentrenamiento"
                )
            with gr.Row():
                pretrain_epochs_input = gr.Number(
                    label="Épocas",
                    value=10,
                    info="Número de épocas para entrenar el modelo"
                )
                pretrain_batch_size_input = gr.Number(
                    label="Tamaño de Lote",
                    value=8,
                    info="Cantidad de muestras procesadas en cada paso de entrenamiento"
                )
                pretrain_lr_input = gr.Number(
                    label="Tasa de Aprendizaje",
                    value=0.001,
                    info="Magnitud de los pasos de actualización de los pesos"
                )
            with gr.Row():
                pretrain_save_every_input = gr.Number(
                    label="Guardar Punto de Control Cada N Épocas (0=desactivado/solo final)",
                    value=0,
                    info="Frecuencia (en épocas) para guardar puntos de control del modelo"
                )
                pretrain_output_dir_input = gr.Textbox(
                    label="Directorio de Salida (opcional, ej: ./pretrain_output)",
                    placeholder="Por defecto en el directorio actual",
                    info="Dónde guardar los modelos preentrenados y puntos de control"
                )
            pretrain_output = gr.Textbox(label="Estado del Preentrenamiento", interactive=False)
            with gr.Row():
                pretrain_btn = gr.Button("Ejecutar Preentrenamiento")
                pretrain_stop_btn = gr.Button("Detener Preentrenamiento")

            pretrain_event = pretrain_btn.click(
                pretrain_interface,
                inputs=[pretrain_dir_input, pretrain_epochs_input, pretrain_batch_size_input, pretrain_lr_input, pretrain_save_every_input, pretrain_output_dir_input],
                outputs=pretrain_output,
                # Add request automatically by Gradio if fn accepts _request or request: gr.Request
            )
            pretrain_stop_btn.click(
                lambda: "Se envió la señal para detener el preentrenamiento. El proceso se detendrá si verifica la cancelación.",
                None,
                pretrain_output,
                cancels=[pretrain_event]
            )

        with gr.TabItem("Fine-tuning"):
            gr.Markdown("## Entrenamiento")
            gr.Markdown("Ingrese la ruta a un directorio que contenga subdirectorios para cada clase, cada uno con archivos de audio. Opcionalmente seleccione un punto de control de modelo para continuar el entrenamiento. Configure los parámetros de entrenamiento.")
            with gr.Row():
                train_dir_input = gr.Textbox(
                    label="Ruta del Directorio de Datos",
                    placeholder="/ruta/a/directorio_datos",
                    info="Directorio con subcarpetas para cada clase, cada una con archivos de audio"
                )
                train_model_input = gr.File(
                    label="Punto de Control del Modelo (.pt)",
                )
            train_classes = gr.Textbox(label="Clases Disponibles", interactive=False)
            train_dir_input.change(list_classes, inputs=train_dir_input, outputs=train_classes)
            with gr.Row():
                epochs_input = gr.Number(
                    label="Épocas",
                    value=10,
                    info="Número de épocas para entrenar el modelo"
                )
                batch_size_input = gr.Number(
                    label="Tamaño de Lote",
                    value=8,
                    info="Cantidad de muestras procesadas en cada paso de entrenamiento"
                )
                lr_input = gr.Number(
                    label="Tasa de Aprendizaje",
                    value=0.001,
                    info="Magnitud de los pasos de actualización de los pesos"
                )
            with gr.Row():
                train_output_dir_input = gr.Textbox(
                    label="Directorio de Salida (opcional, ej: ./train_output)",
                    placeholder="Por defecto en el directorio actual",
                    info="Dónde guardar los modelos entrenados y puntos de control"
                )
            train_output = gr.Textbox(label="Estado y Resultados del Entrenamiento", interactive=False)
            with gr.Row():
                train_btn = gr.Button("Entrenar y Evaluar")
                train_stop_btn = gr.Button("Detener Entrenamiento")

            train_event = train_btn.click(
                train_interface,
                inputs=[train_dir_input, train_model_input, epochs_input, batch_size_input, lr_input, train_output_dir_input],
                outputs=train_output,
                # Add request automatically by Gradio if fn accepts _request or request: gr.Request
            )
            train_stop_btn.click(
                lambda: "Se envió la señal para detener el entrenamiento. El proceso se detendrá si verifica la cancelación.",
                None,
                train_output,
                cancels=[train_event]
            )

        with gr.TabItem("Evaluar"):
            gr.Markdown("## Evaluación")
            gr.Markdown("Ingrese la ruta a un directorio que contenga subdirectorios para cada clase, cada uno con archivos de audio. Seleccione un punto de control de modelo entrenado para evaluar.")
            with gr.Row():
                eval_dir_input = gr.Textbox(
                    label="Ruta del Directorio de Datos",
                    placeholder="/ruta/a/directorio_datos",
                    info="Directorio con subcarpetas para cada clase, cada una con archivos de audio"
                )
                eval_model_input = gr.File(
                    label="Punto de Control del Modelo (.pt)",
                )
            eval_classes = gr.Textbox(label="Clases Disponibles", interactive=False)
            eval_dir_input.change(list_classes, inputs=eval_dir_input, outputs=eval_classes)
            eval_output = gr.Label(label="Conteo de Clases en Evaluación")
            eval_btn = gr.Button("Evaluar")
            eval_btn.click(eval_interface, inputs=[eval_dir_input, eval_model_input], outputs=eval_output)

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
 
        with gr.TabItem("Visualización de Características"):
            gr.Markdown("## Visualización de Características (UMAP & t-SNE)")
            gr.Markdown("Visualice las características de la red colapsadas en 2D con UMAP y t-SNE. Seleccione un directorio de datos etiquetados y un backbone preentrenado.")
            with gr.Row():
                vis_data_dir = gr.Textbox(
                    label="Directorio de Datos Etiquetados",
                    placeholder="/ruta/a/directorio_datos",
                    info="Directorio con subcarpetas para cada clase, cada una con archivos de audio"
                )
                vis_model_path = gr.Textbox(
                    label="Ruta del Modelo Backbone",
                    placeholder="pretrained_backbone.pt",
                    info="Ruta al archivo del backbone preentrenado"
                )
                vis_n_samples = gr.Number(
                    label="Número de Muestras",
                    value=500,
                    info="Cantidad máxima de muestras a visualizar"
                )
            with gr.Row():
                umap_btn = gr.Button("Ejecutar Visualización UMAP")
                tsne_btn = gr.Button("Ejecutar Visualización t-SNE")
            with gr.Row():
                umap_img = gr.Image(type="pil", label="Gráfico UMAP")
                tsne_img = gr.Image(type="pil", label="Gráfico t-SNE")
            def _umap_vis(data_dir, model_path, n_samples):
                result = umap_visualization_interface(data_dir, model_path, int(n_samples))
                if isinstance(result, dict) and "error" in result:
                    return None
                return result
            def _tsne_vis(data_dir, model_path, n_samples):
                result = tsne_visualization_interface(data_dir, model_path, int(n_samples))
                if isinstance(result, dict) and "error" in result:
                    return None
                return result
            umap_btn.click(_umap_vis, inputs=[vis_data_dir, vis_model_path, vis_n_samples], outputs=umap_img)
            tsne_btn.click(_tsne_vis, inputs=[vis_data_dir, vis_model_path, vis_n_samples], outputs=tsne_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
