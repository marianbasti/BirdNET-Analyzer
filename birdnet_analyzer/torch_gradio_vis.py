import gradio as gr
import logging
import os
import numpy as np
import torch
import torchaudio
import umap
import matplotlib.pyplot as plt
import io
import PIL.Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from birdnet_analyzer.torch_model import EfficientNetBackbone, BirdNETMelSpecLayer

# Global cache for extracted features to avoid recomputation when only changing visualization parameters
_feature_cache = {}

def pad_or_truncate(waveform, target_length=2880000):
    """Pad or truncate waveform to fixed length (1 minute = 2,880,000 samples at 48kHz)"""
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
    
    # Load model components
    try:
        logger.debug(f"Loading model components from {model_path}")
        state = torch.load(model_path, map_location=device)
        
        spec_layer = BirdNETMelSpecLayer().to(device)
        spec_layer.eval()
        
        backbone = EfficientNetBackbone(2, 1024).to(device)
        
        if 'backbone.stem.0.weight' in state:
            backbone_state = {k.replace('backbone.', ''): v for k, v in state.items() if k.startswith('backbone.')}
            backbone.load_state_dict(backbone_state, strict=True)
        elif 'stem.0.weight' in state:
            backbone.load_state_dict(state, strict=True)
        else:
            backbone.load_state_dict(state, strict=False)
            
        backbone.eval()
        logger.debug("Model components loaded successfully")
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
    
    features = np.stack(features)
    y_labels = np.array(y_labels)
    
    # Handle NaN/Inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features = features[valid_mask]
        y_labels = y_labels[valid_mask]
    
    # Standardization
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

def create_vis_tab():
    """Create the visualization tab interface."""
    with gr.TabItem("Visualización de Características"):
        gr.Markdown("## Visualización de Características (UMAP & t-SNE)")
        gr.Markdown("Visualice las características de la red colapsadas en 2D con UMAP y t-SNE. Seleccione un directorio de datos etiquetados y un backbone preentrenado. Use los sliders para ajustar los parámetros de visualización en tiempo real.")
        
        with gr.Row():
            with gr.Column(scale=1):
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
                vis_n_samples = gr.Slider(
                    minimum=50, maximum=2000, step=50, value=500,
                    label="Número de Muestras",
                    info="Cantidad máxima de muestras a visualizar"
                )
                
                extract_btn = gr.Button("Extraer Características", variant="primary")
                clear_cache_btn = gr.Button("Limpiar Caché", variant="secondary")
                
                gr.Markdown("### Parámetros UMAP")
                umap_n_neighbors = gr.Slider(
                    minimum=2, maximum=100, step=1, value=15,
                    label="N Neighbors",
                    info="Número de vecinos considerados en el análisis local"
                )
                umap_min_dist = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.1,
                    label="Min Distance",
                    info="Distancia mínima entre puntos en el espacio embebido"
                )
                umap_metric = gr.Dropdown(
                    choices=["cosine", "euclidean", "manhattan", "chebyshev"],
                    value="cosine",
                    label="Métrica de Distancia",
                    info="Métrica utilizada para calcular distancias"
                )
                
                gr.Markdown("### Parámetros t-SNE")
                tsne_perplexity = gr.Slider(
                    minimum=5, maximum=100, step=1, value=30,
                    label="Perplexity",
                    info="Balance entre estructura local y global"
                )
                tsne_n_iter = gr.Slider(
                    minimum=250, maximum=2000, step=250, value=1000,
                    label="Iteraciones",
                    info="Número máximo de iteraciones para optimización"
                )
                tsne_metric = gr.Dropdown(
                    choices=["cosine", "euclidean", "manhattan", "chebyshev"],
                    value="cosine",
                    label="Métrica de Distancia",
                    info="Métrica utilizada para calcular distancias"
                )
            
            with gr.Column(scale=2):
                with gr.Row():
                    umap_img = gr.Image(type="pil", label="Gráfico UMAP")
                    tsne_img = gr.Image(type="pil", label="Gráfico t-SNE")
                
                extraction_status = gr.Textbox(
                    label="Estado de Extracción", 
                    interactive=False,
                    value="Presiona 'Extraer Características' para comenzar"
                )
        
        # Event handlers
        def extract_features_handler(data_dir, model_path, n_samples):
            if not data_dir or not model_path:
                return "Error: Proporciona directorio de datos y ruta del modelo", None, None
            
            result = extract_features_cached(data_dir, model_path, int(n_samples))
            
            if "error" in result:
                return f"Error: {result['error']}", None, None
            
            # Generate initial plots
            umap_img, _ = create_umap_plot(
                result["features"], result["labels"], result["class_names"]
            )
            tsne_img, _ = create_tsne_plot(
                result["features"], result["labels"], result["class_names"]
            )
            
            return f"Características extraídas exitosamente: {result['features'].shape[0]} muestras, {len(result['class_names'])} clases", umap_img, tsne_img
        
        def clear_cache_handler():
            global _feature_cache
            _feature_cache.clear()
            return "Caché limpiado", None, None
        
        # Extract features button
        extract_btn.click(
            extract_features_handler,
            inputs=[vis_data_dir, vis_model_path, vis_n_samples],
            outputs=[extraction_status, umap_img, tsne_img]
        )
        
        # Clear cache button
        clear_cache_btn.click(
            clear_cache_handler,
            outputs=[extraction_status, umap_img, tsne_img]
        )
        
        # UMAP parameter updates
        for component in [umap_n_neighbors, umap_min_dist, umap_metric]:
            component.change(
                update_umap_plot,
                inputs=[vis_data_dir, vis_model_path, vis_n_samples, umap_n_neighbors, umap_min_dist, umap_metric],
                outputs=umap_img
            )
        
        # t-SNE parameter updates  
        for component in [tsne_perplexity, tsne_n_iter, tsne_metric]:
            component.change(
                update_tsne_plot,
                inputs=[vis_data_dir, vis_model_path, vis_n_samples, tsne_perplexity, tsne_n_iter, tsne_metric],
                outputs=tsne_img
            )
