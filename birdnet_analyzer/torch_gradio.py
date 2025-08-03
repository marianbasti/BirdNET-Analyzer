import gradio as gr
# Import modularized logic
from .torch_gradio_classify import classify_interface, list_classes
from .torch_gradio_pretrain import pretrain_interface
from .torch_gradio_train import train_interface
from .torch_gradio_eval import eval_interface
from .torch_gradio_features import (
    extract_features_cached, create_umap_plot, create_tsne_plot,
    update_umap_plot, update_tsne_plot,
    umap_visualization_interface, tsne_visualization_interface,
    _feature_cache
)

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
                pretrain_use_whisper = gr.Checkbox(label="Usar Whisper Backbone", value=True, info="Activar para preentrenar un modelo basado en Whisper")
            
            with gr.Group(visible=True) as whisper_params:
                gr.Markdown("### Parámetros del Backbone Whisper")
                with gr.Row():
                    pretrain_d_model = gr.Number(label="Dimensión del Modelo (d_model)", value=512, info="Dimensión interna del transformer")
                    pretrain_n_heads = gr.Number(label="Número de Cabezas (n_heads)", value=8, info="Número de cabezas de atención")
                    pretrain_n_layers = gr.Number(label="Número de Capas (n_layers)", value=6, info="Número de bloques de transformer")

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
                pretrain_log_every_input = gr.Number(
                    label="Log every (N)",
                    value=1,
                    info="Frecuencia para registrar métricas en wandb (cada N pasos o épocas)"
                )
                pretrain_log_on_input = gr.Dropdown(
                    choices=["epoch", "step"],
                    value="epoch",
                    label="Log on",
                    info="Registrar métricas cada N épocas o pasos"
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

            pretrain_use_whisper.change(lambda x: gr.update(visible=x), inputs=pretrain_use_whisper, outputs=whisper_params)

            pretrain_event = pretrain_btn.click(
                pretrain_interface,
                inputs=[
                    pretrain_dir_input, pretrain_epochs_input, pretrain_batch_size_input, 
                    pretrain_lr_input, pretrain_save_every_input, pretrain_output_dir_input,
                    pretrain_use_whisper, pretrain_d_model, pretrain_n_heads, pretrain_n_layers,
                    pretrain_log_every_input, pretrain_log_on_input
                ],
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
                        placeholder="pretrained_backbone_complete.pt",
                        info="Ruta al archivo del backbone preentrenado (ej. whisper_pretrained_backbone_complete.pt)"
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
