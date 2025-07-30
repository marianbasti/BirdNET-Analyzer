import logging
import numpy as np
import umap
import matplotlib.pyplot as plt
import io
import torch
import torchaudio
import PIL.Image
from mpl_toolkits.mplot3d import Axes3D  # <-- add for 3D plotting
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from birdnet_analyzer.torch_utils import pad_or_truncate
from birdnet_analyzer.torch_model import DeltaNet

_feature_cache = {}

def extract_features_cached(data_dir, model_path, n_samples=500):
    """Extract and cache features to avoid recomputation when only changing visualization parameters."""
    import os

    cache_key = f"{data_dir}_{model_path}_{n_samples}"
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

    try:
        logger.debug(f"Loading model from {model_path}")
        state = torch.load(model_path, map_location=device)
        num_classes = state['classifier.weight'].shape[0] if 'classifier.weight' in state else 10
        model = DeltaNet(num_classes=num_classes).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}

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

    if len(audio_paths) > n_samples:
        idxs = np.random.choice(len(audio_paths), n_samples, replace=False)
        audio_paths = [audio_paths[i] for i in idxs]
        labels = [labels[i] for i in idxs]

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
                o = model.spec_layer(waveform)
                B, C, F, T = o.shape
                mel_seq = o.permute(0, 3, 1, 2).reshape(B, T, C * F)
                q_in = model.q_proj(mel_seq)
                k_in = model.k_proj(mel_seq)
                v_in = model.v_proj(mel_seq)
                q_in, _ = model.q_conv1d(q_in)
                k_in, _ = model.k_conv1d(k_in)
                v_in, _ = model.v_conv1d(v_in)
                q = q_in
                pooled = q.mean(dim=1)
                emb_normalized = torch.nn.functional.normalize(pooled, dim=1)
                features.append(emb_normalized.cpu().numpy()[0])
                y_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                continue

    if not features:
        return {"error": "No se extrajeron características."}

    features = np.stack(features)
    y_labels = np.array(y_labels)

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features = features[valid_mask]
        y_labels = y_labels[valid_mask]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    result = {
        "features": features_scaled,
        "labels": y_labels,
        "class_names": class_names,
        "success": True
    }
    _feature_cache[cache_key] = result

    logger.debug(f"Features extracted and cached: {features_scaled.shape}")
    return result

def create_umap_plot(features, labels, class_names, n_neighbors=15, min_dist=0.1, metric='cosine', n_components=2):
    try:
        reducer = umap.UMAP(
            n_components=int(n_components),
            random_state=42,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=metric
        )
        embedding = reducer.fit_transform(features)
    except Exception as e:
        return None, f"UMAP error: {e}"

    if int(n_components) == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, cname in enumerate(class_names):
        mask = labels == idx
        if np.any(mask):
            if int(n_components) == 3:
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                    label=cname, alpha=0.7, s=30, c=[colors[idx]]
                )
            else:
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=cname, alpha=0.7, s=30, c=[colors[idx]]
                )
    ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})', fontsize=14)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    if int(n_components) == 3:
        ax.set_zlabel('UMAP 3', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img, None

def create_tsne_plot(features, labels, class_names, perplexity=30, n_iter=1000, metric='cosine', n_components=2):
    try:
        tsne = TSNE(
            n_components=int(n_components),
            random_state=42,
            init='pca',
            perplexity=min(int(perplexity), len(features)//4),
            n_iter=int(n_iter),
            metric=metric
        )
        embedding = tsne.fit_transform(features)
    except Exception as e:
        return None, f"t-SNE error: {e}"

    if int(n_components) == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, cname in enumerate(class_names):
        mask = labels == idx
        if np.any(mask):
            if int(n_components) == 3:
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                    label=cname, alpha=0.7, s=30, c=[colors[idx]]
                )
            else:
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=cname, alpha=0.7, s=30, c=[colors[idx]]
                )
    ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f't-SNE (perplexity={perplexity}, n_iter={n_iter}, metric={metric})', fontsize=14)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    if int(n_components) == 3:
        ax.set_zlabel('t-SNE 3', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img, None

def update_umap_plot(data_dir, model_path, n_samples, n_neighbors, min_dist, metric, n_components=2):
    feature_data = extract_features_cached(data_dir, model_path, int(n_samples))
    if "error" in feature_data:
        return None
    if not feature_data.get("success", False):
        return None
    img, error = create_umap_plot(
        feature_data["features"],
        feature_data["labels"],
        feature_data["class_names"],
        n_neighbors, min_dist, metric, n_components
    )
    return img

def update_tsne_plot(data_dir, model_path, n_samples, perplexity, n_iter, metric, n_components=2):
    feature_data = extract_features_cached(data_dir, model_path, int(n_samples))
    if "error" in feature_data:
        return None
    if not feature_data.get("success", False):
        return None
    img, error = create_tsne_plot(
        feature_data["features"],
        feature_data["labels"],
        feature_data["class_names"],
        perplexity, n_iter, metric, n_components
    )
    return img

def umap_visualization_interface(data_dir, model_path, n_samples=500, n_components=2):
    feature_data = extract_features_cached(data_dir, model_path, n_samples)
    if "error" in feature_data:
        return feature_data
    img, error = create_umap_plot(
        feature_data["features"],
        feature_data["labels"],
        feature_data["class_names"],
        n_components=n_components
    )
    return img if img else {"error": error}

def tsne_visualization_interface(data_dir, model_path, n_samples=500, n_components=2):
    feature_data = extract_features_cached(data_dir, model_path, n_samples)
    if "error" in feature_data:
        return feature_data
    img, error = create_tsne_plot(
        feature_data["features"],
        feature_data["labels"],
        feature_data["class_names"],
        n_components=n_components
    )
    return img if img else {"error": error}
