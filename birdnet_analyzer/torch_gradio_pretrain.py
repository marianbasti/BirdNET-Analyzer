import logging
from sklearn.manifold import TSNE
from birdnet_analyzer.torch_utils import pad_or_truncate, list_classes
from birdnet_analyzer.torch_train_utils import _infer_hidden_size_from_backbone_state


def classify_interface(audio_dir, model_path, window_size_sec, hop_size_sec, threshold=0.5):
    import os
    import torch
    import torchaudio
    from birdnet_analyzer.torch_model import DeltaNet
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
        model = DeltaNet(num_classes=num_classes)
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

import numpy as np
import umap
import matplotlib.pyplot as plt
import io
import gradio as gr
import torch
import torchaudio
from birdnet_analyzer.torch_model import DeltaNet
from birdnet_analyzer.torch_pretrain_utils import SimCLRPretrainer, UnlabeledAudioDataset, collate_fn

# Add request: gr.Request to function signature
def pretrain_interface(
    data_dir,
    epochs,
    batch_size,
    learning_rate,
    save_every_epochs=0,
    output_dir=None,
    progress=None,
    request=None,
    hidden_size=192,
    proj_dim=128
):
    import os
    from torch.utils.data import DataLoader

    # Debug: print the received data_dir
    print(f"[DEBUG] pretrain_interface received data_dir: '{data_dir}'")

    if not data_dir or not os.path.isdir(data_dir):
        return {"error": f"Por favor, proporciona una ruta de directorio válida. Recibido: '{data_dir}'"}
    try:
        dataset = UnlabeledAudioDataset(data_dir)
        if len(dataset) == 0:
            return {"error": f"No se encontraron archivos de audio en el directorio proporcionado: '{data_dir}'"}
        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, collate_fn=collate_fn)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # --- Pass hidden_size and proj_dim to SimCLRPretrainer ---
        pretrainer = SimCLRPretrainer(
            emb_size=int(hidden_size),
            proj_dim=int(proj_dim),
            device=device
        )
        # Handle output_dir
        if output_dir is not None and output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'pretrained_backbone.pt')
        else:
            save_path = 'pretrained_backbone.pt'
        # Remove checkpoint_prefix argument
        pretrainer.train(
            dataloader,
            epochs=int(epochs),
            lr=float(learning_rate),
            save_path=save_path,
            checkpoint_every=int(save_every_epochs)
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": f"Error en el preentrenamiento: {e}\nTraceback:\n{tb}"}

# Add request: gr.Request to function signature
def train_interface(data_dir, model_path, epochs, batch_size, learning_rate, output_dir=None, progress=None, request=None):
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from birdnet_analyzer.torch_train_utils import AudioDataset, train_model
    import torchaudio
    import logging
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
        model = None
        if model_path is not None:
            try:
                state = torch.load(model_path, map_location=device)
                # --- Robust checkpoint loading with hidden_size adaptation ---
                if isinstance(state, dict) and "backbone" in state:
                    backbone_state = state["backbone"]
                    # --- Infer hidden_size from backbone_state ---
                    hidden_size = _infer_hidden_size_from_backbone_state(backbone_state)
                    model = DeltaNet(num_classes=len(class_names), hidden_size=hidden_size)
                    # Remove classifier weights if present in backbone_state
                    backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith("classifier.")}
                    model.load_state_dict(backbone_state, strict=False)
                    # Replace classifier for correct output dim
                    model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names)).to(device)
                    print(f"Se cargaron los pesos del backbone para el fine-tuning (SimCLR checkpoint, hidden_size={hidden_size}).")
                elif isinstance(state, dict) and any(k.startswith("classifier.") for k in state.keys()):
                    # --- Infer hidden_size from state dict if possible ---
                    hidden_size = None
                    if "q_proj.weight" in state:
                        hidden_size = state["q_proj.weight"].shape[0]
                    elif "v_proj.weight" in state:
                        hidden_size = state["v_proj.weight"].shape[0]
                    elif "o_proj.weight" in state:
                        hidden_size = state["o_proj.weight"].shape[0]
                    model = DeltaNet(num_classes=len(class_names), hidden_size=hidden_size) if hidden_size else DeltaNet(num_classes=len(class_names))
                    try:
                        model.load_state_dict(state, strict=True)
                    except RuntimeError as e:
                        print(f"Advertencia: {e}")
                        # Remove classifier weights from state dict before loading
                        filtered_state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
                        model.load_state_dict(filtered_state, strict=False)
                        model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names)).to(device)
                        print("Se re-inicializó la capa de clasificación para el número correcto de clases.")
                else:
                    model = DeltaNet(num_classes=len(class_names))
                    model.load_state_dict(state, strict=False)
                    print("Advertencia: Formato de checkpoint desconocido, se cargaron los pesos coincidentes.")
            except Exception as e:
                return {"error": f"Error al cargar el modelo: {e}"}
        else:
            model = DeltaNet(num_classes=len(class_names))
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
    import logging
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
        model = None
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "backbone" in state:
            backbone_state = state["backbone"]
            hidden_size = _infer_hidden_size_from_backbone_state(backbone_state)
            model = DeltaNet(num_classes=len(class_names), hidden_size=hidden_size)
            backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith("classifier.")}
            model.load_state_dict(backbone_state, strict=False)
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names)).to(device)
            print(f"Se cargaron los pesos del backbone para la evaluación (SimCLR checkpoint, hidden_size={hidden_size}).")
        elif isinstance(state, dict) and any(k.startswith("classifier.") for k in state.keys()):
            model = DeltaNet(num_classes=len(class_names))
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as e:
                print(f"Advertencia: {e}")
                filtered_state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
                model.load_state_dict(filtered_state, strict=False)
                model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names)).to(device)
                print("Se re-inicializó la capa de clasificación para el número correcto de clases.")
        else:
            model = DeltaNet(num_classes=len(class_names))
            model.load_state_dict(state, strict=False)
            print("Advertencia: Formato de checkpoint desconocido, se cargaron los pesos coincidentes.")
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
# Global cache for extracted features to avoid recomputation when only changing visualization parameters
_feature_cache = {}

def extract_features_cached(data_dir, model_path, n_samples=500):
    """Extract and cache features to avoid recomputation when only changing visualization parameters."""
    import torch
    import torchaudio
    from birdnet_analyzer.torch_model import DeltaNet
    import os
    import numpy as np
    
    # Create cache key
    cache_key = f"{data_dir}_{model_path}_{n_samples}"
    
    # Return cached result if available
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("feature_extraction")

    logger.debug(f"Extracting features for: {data_dir}")
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model (use full DeltaNet, not just backbone)
    try:
        logger.debug(f"Loading model from {model_path}")
        state = torch.load(model_path, map_location=device)
        # Infer number of classes from state dict if possible
        num_classes = state['classifier.weight'].shape[0] if 'classifier.weight' in state else 10
        model = DeltaNet(num_classes=num_classes).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}

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
        import numpy as np
        idxs = np.random.choice(len(audio_paths), n_samples, replace=False)
        audio_paths = [audio_paths[i] for i in idxs]
        labels = [labels[i] for i in idxs]

    # Extract features
    features = []
    y_labels = []
    
    with torch.no_grad():
        for path, label in zip(audio_paths, labels):
            try:
                waveform, sr = torchaudio.load(path)
                if sr != 48000:
                    waveform = torchaudio.functional.resample(waveform, sr, 48000)
                if waveform.ndim > 1:
                    waveform = waveform[0]
                waveform = pad_or_truncate(waveform, 2880000)
                waveform = waveform.unsqueeze(0).to(device)
                # Forward through full model, get penultimate layer
                # Get features before classifier
                o = model.spec_layer(waveform)  # (B, 2, 96, T)
                B, C, F, T = o.shape
                mel_seq = o.permute(0, 3, 1, 2).reshape(B, T, C * F)
                q_in = model.q_proj(mel_seq)
                k_in = model.k_proj(mel_seq)
                v_in = model.v_proj(mel_seq)
                q_in, _ = model.q_conv1d(q_in)
                k_in, _ = model.k_conv1d(k_in)
                v_in, _ = model.v_conv1d(v_in)
                q = q_in
                # Pool over time axis (mean pooling)
                pooled = q.mean(dim=1)
                emb_normalized = torch.nn.functional.normalize(pooled, dim=1)
                features.append(emb_normalized.cpu().numpy()[0])
                y_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                continue
    
    if not features:
        return {"error": "No se extrajeron características."}
    
    import numpy as np
    features = np.stack(features)
    y_labels = np.array(y_labels)
    
    # Handle NaN/Inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features = features[valid_mask]
        y_labels = y_labels[valid_mask]
    
    # Standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cache the result
    result = {
        "features": features_scaled,
        "labels": y_labels,
        "class_names": class_names,
        "success": True
    }
    _feature_cache[cache_key] = result
    
    logger.debug(f"Features extracted and cached: {features_scaled.shape}")
    return result

def create_umap_plot(features, labels, class_names, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """Create UMAP plot with given parameters."""
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import PIL.Image
    
    try:
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=metric
        )
        embedding = reducer.fit_transform(features)
    except Exception as e:
        return None, f"UMAP error: {e}"

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for idx, cname in enumerate(class_names):
        mask = labels == idx
        if np.any(mask):
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      label=cname, alpha=0.7, s=30, c=[colors[idx]])
    
    ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})', fontsize=14)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img, None

def create_tsne_plot(features, labels, class_names, perplexity=30, n_iter=1000, metric='cosine'):
    """Create t-SNE plot with given parameters."""
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import PIL.Image
    from sklearn.manifold import TSNE
    
    try:
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            init='pca',
            perplexity=min(int(perplexity), len(features)//4),
            n_iter=int(n_iter),
            metric=metric
        )
        embedding = tsne.fit_transform(features)
    except Exception as e:
        return None, f"t-SNE error: {e}"

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for idx, cname in enumerate(class_names):
        mask = labels == idx
        if np.any(mask):
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      label=cname, alpha=0.7, s=30, c=[colors[idx]])
    
    ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f't-SNE (perplexity={perplexity}, n_iter={n_iter}, metric={metric})', fontsize=14)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img, None

def update_umap_plot(data_dir, model_path, n_samples, n_neighbors, min_dist, metric):
    """Update UMAP plot with new parameters."""
    feature_data = extract_features_cached(data_dir, model_path, int(n_samples))
    
    if "error" in feature_data:
        return None
    
    if not feature_data.get("success", False):
        return None
        
    img, error = create_umap_plot(
        feature_data["features"], 
        feature_data["labels"], 
        feature_data["class_names"],
        n_neighbors, min_dist, metric
    )
    
    return img

def update_tsne_plot(data_dir, model_path, n_samples, perplexity, n_iter, metric):
    """Update t-SNE plot with new parameters."""
    feature_data = extract_features_cached(data_dir, model_path, int(n_samples))
    
    if "error" in feature_data:
        return None
    
    if not feature_data.get("success", False):
        return None
        
    img, error = create_tsne_plot(
        feature_data["features"], 
        feature_data["labels"], 
        feature_data["class_names"],
        perplexity, n_iter, metric
    )
    
    return img

# Keep original functions for backward compatibility
def umap_visualization_interface(data_dir, model_path, n_samples=500):
    feature_data = extract_features_cached(data_dir, model_path, n_samples)
    if "error" in feature_data:
        return feature_data
    
    img, error = create_umap_plot(
        feature_data["features"], 
        feature_data["labels"], 
        feature_data["class_names"]
    )
    return img if img else {"error": error}

def tsne_visualization_interface(data_dir, model_path, n_samples=500):
    feature_data = extract_features_cached(data_dir, model_path, n_samples)
    if "error" in feature_data:
        return feature_data
    
    img, error = create_tsne_plot(
        feature_data["features"], 
        feature_data["labels"], 
        feature_data["class_names"]
    )
    return img if img else {"error": error}

if __name__ == "__main__":
    from birdnet_analyzer.gradio_ui import demo
    demo.launch(server_name="0.0.0.0")