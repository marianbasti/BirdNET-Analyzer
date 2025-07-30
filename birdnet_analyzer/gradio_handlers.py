from birdnet_analyzer.torch_gradio import (
    extract_features_cached,
    create_umap_plot,
    create_tsne_plot,
)

def extract_features_handler(data_dir, model_path, n_samples):
    if not data_dir or not model_path:
        return "Error: Proporciona directorio de datos y ruta del modelo", None, None

    result = extract_features_cached(data_dir, model_path, int(n_samples))

    if "error" in result:
        return f"Error: {result['error']}", None, None

    umap_img, _ = create_umap_plot(
        result["features"], result["labels"], result["class_names"]
    )
    tsne_img, _ = create_tsne_plot(
        result["features"], result["labels"], result["class_names"]
    )

    return f"Características extraídas exitosamente: {result['features'].shape[0]} muestras, {len(result['class_names'])} clases", umap_img, tsne_img

def clear_cache_handler():
    from birdnet_analyzer.torch_gradio import _feature_cache
    _feature_cache.clear()
    return "Caché limpiado", None, None
