import gradio as gr
from birdnet_analyzer.torch_gradio import (
    pretrain_interface,
    train_interface,
    eval_interface,
    classify_interface,
    list_classes,
    update_umap_plot,
    update_tsne_plot,
)
from birdnet_analyzer.gradio_handlers import extract_features_handler

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
                pretrain_hidden_size_input = gr.Number(
                    label="Dimensión del Backbone",
                    value=192,
                    info="Tamaño de la capa oculta principal (hidden_size)"
                )
                pretrain_proj_dim_input = gr.Number(
                    label="Dimensión de Proyección",
                    value=128,
                    info="Tamaño de la proyección SimCLR (proj_dim)"
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
                inputs=[
                    pretrain_dir_input,
                    pretrain_epochs_input,
                    pretrain_batch_size_input,
                    pretrain_lr_input,
                    pretrain_save_every_input,
                    pretrain_output_dir_input,
                    pretrain_hidden_size_input,
                    pretrain_proj_dim_input
                ],
                outputs=pretrain_output,
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
            train_model_dims = gr.Textbox(
                label="Dimensiones del Backbone y Proyección (del checkpoint)",
                interactive=False,
                value="Cargue un checkpoint para ver dimensiones"
            )
            train_classes = gr.Textbox(label="Clases Disponibles", interactive=False)
            train_dir_input.change(list_classes, inputs=train_dir_input, outputs=train_classes)
            # --- Update model dims when checkpoint is selected ---
            def show_model_dims(model_file):
                import torch
                if not model_file:
                    return "Cargue un checkpoint para ver dimensiones"
                try:
                    state = torch.load(model_file.name, map_location="cpu")
                    # Backbone dims
                    if "backbone" in state:
                        backbone = state["backbone"]
                        if "q_proj.weight" in backbone:
                            hidden_size = backbone["q_proj.weight"].shape[0]
                        elif "v_proj.weight" in backbone:
                            hidden_size = backbone["v_proj.weight"].shape[0]
                        elif "o_proj.weight" in backbone:
                            hidden_size = backbone["o_proj.weight"].shape[0]
                        else:
                            hidden_size = "?"
                        proj_dim = None
                        if "proj_head" in state:
                            proj = state["proj_head"]
                            # Try to get last Linear out_features
                            for k, v in proj.items():
                                if ".weight" in k and v.ndim == 2:
                                    proj_dim = v.shape[0]
                        return f"Backbone hidden_size: {hidden_size} | Proj dim: {proj_dim if proj_dim else 'N/A'}"
                    # Direct state dict
                    if "q_proj.weight" in state:
                        hidden_size = state["q_proj.weight"].shape[0]
                    elif "v_proj.weight" in state:
                        hidden_size = state["v_proj.weight"].shape[0]
                    elif "o_proj.weight" in state:
                        hidden_size = state["o_proj.weight"].shape[0]
                    else:
                        hidden_size = "?"
                    return f"Backbone hidden_size: {hidden_size}"
                except Exception as e:
                    return f"Error leyendo dimensiones: {e}"
            train_model_input.change(
                show_model_dims,
                inputs=train_model_input,
                outputs=train_model_dims
            )
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
                    
                    # Add dropdown for 2D/3D
                    vis_n_components = gr.Dropdown(
                        choices=[2, 3],
                        value=2,
                        label="Dimensiones de Visualización",
                        info="Selecciona 2D o 3D para la visualización"
                    )
                    
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
            extract_btn.click(
                extract_features_handler,
                inputs=[vis_data_dir, vis_model_path, vis_n_samples],
                outputs=[extraction_status, umap_img, tsne_img]
            )
            
            # UMAP parameter updates
            for component in [umap_n_neighbors, umap_min_dist, umap_metric, vis_n_components]:
                component.change(
                    update_umap_plot,
                    inputs=[vis_data_dir, vis_model_path, vis_n_samples, umap_n_neighbors, umap_min_dist, umap_metric, vis_n_components],
                    outputs=umap_img
                )
            
            # t-SNE parameter updates  
            for component in [tsne_perplexity, tsne_n_iter, tsne_metric, vis_n_components]:
                component.change(
                    update_tsne_plot,
                    inputs=[vis_data_dir, vis_model_path, vis_n_samples, tsne_perplexity, tsne_n_iter, tsne_metric, vis_n_components],
                    outputs=tsne_img
                )

def _infer_hidden_size_from_backbone_state(backbone_state):
    """
    Infer the hidden_size used in the backbone checkpoint.
    Looks for q_proj.weight or v_proj.weight or o_proj.weight.
    """
    if "q_proj.weight" in backbone_state:
        return backbone_state["q_proj.weight"].shape[0]
    for k in ["v_proj.weight", "o_proj.weight"]:
        if k in backbone_state:
            return backbone_state[k].shape[0]
    # Fallback default
    return 192

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")