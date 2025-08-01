import logging
import torch
import torchaudio
import numpy as np
from birdnet_analyzer.torch_model import WhisperBackbone, BirdNETMelSpecLayer

_feature_cache = {}

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


def extract_features_cached(data_dir, model_path, n_samples=500):
    """Extract and cache features to avoid recomputation when only changing visualization parameters."""
    import os
    import numpy as np
    
    # Create cache key
    cache_key = f"{data_dir}_{model_path}_{n_samples}"
    
    # Return cached result if available
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("feature_extraction")

    logger.debug(f"Extracting features for: {data_dir}")
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model components
    try:
        logger.debug(f"Loading model components from {model_path}")
        state = torch.load(model_path, map_location=device)
        
        # Check for config to decide which model to load
        config = state.get('config', {})
        is_whisper = 'd_model' in config or 'n_mels' in config

        if is_whisper:
            spec_layer = BirdNETMelSpecLayer(n_mels=config.get('n_mels', 80)).to(device)
            backbone = WhisperBackbone(
                n_mels=config.get('n_mels', 80),
                d_model=config.get('d_model', 512),
                n_heads=config.get('n_heads', 8),
                n_layers=config.get('n_layers', 6),
                emb_size=config.get('emb_size', 1024)
            ).to(device)
        else:
            # Fallback for old EfficientNet models
            from birdnet_analyzer.torch_model import EfficientNetBackbone
            spec_layer = BirdNETMelSpecLayer(spec_shape=(96, 511)).to(device) # Old spec layer
            backbone = EfficientNetBackbone(2, 1024).to(device)

        spec_layer.eval()
        
        # Load weights
        if 'spec_layer' in state:
            spec_layer.load_state_dict(state['spec_layer'])
        
        if 'backbone' in state:
            backbone.load_state_dict(state['backbone'])
        else:
            # Fallback for older checkpoints that are just the backbone state_dict
            backbone.load_state_dict(state, strict=False)
            
        backbone.eval()
        logger.debug("Model components loaded successfully")
    except Exception as e:
        import traceback
        logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
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
                
                spec = spec_layer(waveform)
                emb = backbone(spec)
                emb_normalized = torch.nn.functional.normalize(emb, dim=1)
                
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