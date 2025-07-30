import logging
from sklearn.manifold import TSNE
from birdnet_analyzer.torch_utils import pad_or_truncate, list_classes

from birdnet_analyzer.torch_gradio_classify import classify_interface
from birdnet_analyzer.torch_gradio_pretrain import pretrain_interface
from birdnet_analyzer.torch_gradio_train import train_interface
from birdnet_analyzer.torch_gradio_eval import eval_interface
from birdnet_analyzer.torch_gradio_visualize import (
    extract_features_cached,
    create_umap_plot,
    create_tsne_plot,
    update_umap_plot,
    update_tsne_plot,
    umap_visualization_interface,
    tsne_visualization_interface,
)

if __name__ == "__main__":
    from birdnet_analyzer.gradio_ui import demo
    demo.launch(server_name="0.0.0.0")
