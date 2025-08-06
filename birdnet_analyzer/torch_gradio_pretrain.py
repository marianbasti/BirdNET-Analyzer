import gradio as gr
import os
import torch
from torch.utils.data import DataLoader
from birdnet_analyzer.torch_pretrain_utils import SimCLRPretrainer, UnlabeledAudioDataset, collate_fn

def get_available_devices():
    """Get list of available CUDA devices"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices

def pretrain_interface(data_dir, epochs, batch_size, learning_rate, device_selection, save_every_epochs=0, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None):
    """Pretraining interface for self-supervised learning with SimCLR."""
    # Debug: print the received data_dir
    print(f"[DEBUG] pretrain_interface received data_dir: '{data_dir}'")

    if not data_dir or not os.path.isdir(data_dir):
        return {"error": f"Por favor, proporciona una ruta de directorio válida. Recibido: '{data_dir}'"}
    
    try:
        dataset = UnlabeledAudioDataset(data_dir)
        if len(dataset) == 0:
            return {"error": f"No se encontraron archivos de audio en el directorio proporcionado: '{data_dir}'"}
        
        print(f"[INFO] Found {len(dataset)} audio files in dataset")
        
        # Test first few samples to ensure they work
        print("[INFO] Testing first few samples...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"[INFO] Sample {i}: shapes {sample[0].shape}, {sample[1].shape}")
            except Exception as e:
                print(f"[WARNING] Error in sample {i}: {e}")
        
        # Reduce batch size and disable multiprocessing to avoid issues
        effective_batch_size = min(int(batch_size), 2)  # Cap batch size at 2 for stability
        
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing to avoid worker issues
            pin_memory=False,  # Disable pin_memory for CPU debugging
            prefetch_factor=None  # Not used when num_workers=0
        )
        
        device = device_selection if device_selection else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")
        
        pretrainer = SimCLRPretrainer(device=device)
        
        # Handle output_dir
        if output_dir is not None and output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'pretrained_backbone.pt')
            checkpoint_prefix = os.path.join(output_dir, 'checkpoint_pretrain_epoch')
        else:
            save_path = 'pretrained_backbone.pt'
            checkpoint_prefix = 'checkpoint_pretrain_epoch'
        
        print(f"[INFO] Starting pretraining with effective batch size: {effective_batch_size}")
        
        # Try to call with checkpoint_prefix if supported
        try:
            pretrainer.train(
                dataloader,
                epochs=int(epochs),
                lr=float(learning_rate),
                save_path=save_path,
                checkpoint_every=int(save_every_epochs),
                use_amp=False  # Disable mixed precision to avoid potential NaN issues
            )
        except TypeError:
            pretrainer.train(dataloader, epochs=int(epochs), lr=float(learning_rate), save_path=save_path, checkpoint_every=int(save_every_epochs))
        
        return {"success": f"Preentrenamiento completado. Modelo guardado en: {save_path}"}
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": f"Error en el preentrenamiento: {e}\nTraceback:\n{tb}"}

def create_pretrain_tab():
    """Create the pretraining tab interface."""
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
            pretrain_device_input = gr.Dropdown(
                label="Dispositivo de Entrenamiento",
                choices=get_available_devices(),
                value=get_available_devices()[0],
                info="Seleccione el dispositivo para entrenamiento (CPU o GPU específica)"
            )
            pretrain_save_every_input = gr.Number(
                label="Guardar Punto de Control Cada N Épocas (0=desactivado/solo final)",
                value=0,
                info="Frecuencia (en épocas) para guardar puntos de control del modelo"
            )
        
        with gr.Row():
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
            inputs=[pretrain_dir_input, pretrain_epochs_input, pretrain_batch_size_input, pretrain_lr_input, pretrain_device_input, pretrain_save_every_input, pretrain_output_dir_input],
            outputs=pretrain_output,
        )
        pretrain_stop_btn.click(
            lambda: "Se envió la señal para detener el preentrenamiento. El proceso se detendrá si verifica la cancelación.",
            None,
            pretrain_output,
            cancels=[pretrain_event]
        )
