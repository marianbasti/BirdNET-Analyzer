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
    devices.append("batch")  # Add batch option for multi-device training
    return devices

def pretrain_interface(data_dir, epochs, batch_size, learning_rate, device_selection, save_every_epochs=0, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None):
    """Pretraining interface for self-supervised learning with SimCLR."""
    # Debug: print the received data_dir
    print(f"[DEBUG] pretrain_interface received data_dir: '{data_dir}'")

    # Accept comma-separated list of directories
    if not data_dir:
        return {"error": f"Por favor, proporciona una ruta de directorio válida. Recibido: '{data_dir}'"}
    dir_list = [d.strip() for d in data_dir.split(",") if d.strip()]
    invalid_dirs = [d for d in dir_list if not os.path.isdir(d)]
    if not dir_list or invalid_dirs:
        return {"error": f"Por favor, proporciona rutas de directorio válidas. Recibido: '{data_dir}'. Directorios inválidos: {invalid_dirs}"}

    try:
        # Combine datasets from all directories
        datasets = []
        for d in dir_list:
            ds = UnlabeledAudioDataset(d)
            if len(ds) > 0:
                datasets.append(ds)
        if not datasets or sum(len(ds) for ds in datasets) == 0:
            return {"error": f"No se encontraron archivos de audio en los directorios proporcionados: {dir_list}"}
        from torch.utils.data import ConcatDataset
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        print(f"[INFO] Found {len(dataset)} audio files in dataset(s)")

        # Test first few samples to ensure they work
        print("[INFO] Testing first few samples...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"[INFO] Sample {i}: shapes {sample[0].shape}, {sample[1].shape}")
            except Exception as e:
                print(f"[WARNING] Error in sample {i}: {e}")

        # Reduce batch size and disable multiprocessing to avoid issues
        effective_batch_size = min(int(batch_size), 128)  # Cap batch size at 2 for stability
        
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

def eval_pretrained_backbone_interface(backbone_path):
    """Evaluate a pretrained backbone: show structure and parameter count."""
    import torch
    import os
    from birdnet_analyzer.torch_model import EfficientNetBackbone

    if not backbone_path or not os.path.isfile(backbone_path):
        return "Por favor, proporciona una ruta válida al archivo del backbone preentrenado (.pt)."

    try:
        emb_size = 1024
        in_ch = 2
        model = EfficientNetBackbone(in_ch=in_ch, emb_size=emb_size)
        state_dict = torch.load(backbone_path, map_location="cpu")
        # Handle both backbone-only and full SimCLR checkpoints
        if isinstance(state_dict, dict) and "backbone" in state_dict:
            # Full SimCLR checkpoint
            model.load_state_dict(state_dict["backbone"])
        elif isinstance(state_dict, dict) and all(k.startswith(("stem", "blocks", "head")) for k in state_dict.keys()):
            # Backbone-only checkpoint
            model.load_state_dict(state_dict)
        else:
            return (
                "El archivo no parece ser un checkpoint de backbone válido. "
                "Asegúrate de seleccionar un archivo .pt guardado como backbone o como checkpoint completo de preentrenamiento."
            )
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary = f"Modelo EfficientNetBackbone cargado.\nTotal parámetros: {total_params:,}\nParámetros entrenables: {trainable_params:,}\n"
        summary += f"Estructura del modelo:\n{model}"
        return summary
    except Exception as e:
        import traceback
        return f"Error al cargar o analizar el backbone: {e}\n{traceback.format_exc()}"

def pretrain_batch_interface(data_dir, epochs, batch_size, learning_rate, save_every_epochs, output_dir, device_configs, progress=gr.Progress(track_tqdm=True)):
    """Pretraining interface for batch training across multiple devices."""
    if not data_dir:
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    
    dir_list = [d.strip() for d in data_dir.split(",") if d.strip()]
    invalid_dirs = [d for d in dir_list if not os.path.isdir(d)]
    if not dir_list or invalid_dirs:
        return {"error": f"Por favor, proporciona rutas de directorio válidas. Directorios inválidos: {invalid_dirs}"}

    try:
        # Combine datasets from all directories
        datasets = []
        for d in dir_list:
            ds = UnlabeledAudioDataset(d)
            if len(ds) > 0:
                datasets.append(ds)
        if not datasets or sum(len(ds) for ds in datasets) == 0:
            return {"error": f"No se encontraron archivos de audio en los directorios proporcionados: {dir_list}"}
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)

        # Launch training for each device
        results = {}
        for device, config in device_configs.items():
            device_output_dir = os.path.join(output_dir, f"pretrain_{device}")
            os.makedirs(device_output_dir, exist_ok=True)
            save_path = os.path.join(device_output_dir, 'pretrained_backbone.pt')
            checkpoint_prefix = os.path.join(device_output_dir, 'checkpoint_pretrain_epoch')

            dataloader = DataLoader(
                dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False
            )

            pretrainer = SimCLRPretrainer(device=device)
            try:
                pretrainer.train(
                    dataloader,
                    epochs=int(config["epochs"]),
                    lr=float(config["learning_rate"]),
                    save_path=save_path,
                    checkpoint_every=int(config["save_every_epochs"]),
                    output_dir=device_output_dir
                )
                results[device] = f"Preentrenamiento completado en {device}. Modelo guardado en: {save_path}"
            except Exception as e:
                results[device] = f"Error en el preentrenamiento en {device}: {e}"

        return results

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": f"Error en el preentrenamiento: {e}\nTraceback:\n{tb}"}

def create_pretrain_tab():
    """Create the pretraining tab interface."""
    with gr.TabItem("Preentrenar"):
        gr.Markdown("## Preentrenamiento Auto-supervisado (SimCLR)")
        gr.Markdown("Ingrese la ruta a uno o más directorios (separados por coma) que contengan archivos de audio no etiquetados (cualquier estructura de carpetas). Configure los parámetros de preentrenamiento. El backbone se guardará como 'pretrained_backbone.pt'.")
        with gr.Row():
            pretrain_dir_input = gr.Textbox(
                label="Ruta(s) del Directorio de Datos No Etiquetados (separadas por coma)",
                placeholder="/ruta/a/datos1, /ruta/a/datos2",
                info="Uno o más directorios (separados por coma) que contienen archivos de audio no etiquetados para preentrenamiento"
            )
        with gr.Row():
            pretrain_device_input = gr.Dropdown(
                label="Dispositivo de Entrenamiento",
                choices=get_available_devices(),
                value=get_available_devices()[0],
                info="Seleccione el dispositivo para entrenamiento (CPU, GPU específica o batch para múltiples GPUs)"
            )
        pretrain_tabs = gr.Tabs()
        device_configs = {}

        with pretrain_tabs:
            for device in get_available_devices():
                if device == "batch":
                    continue
                with gr.TabItem(f"Parámetros para {device}"):
                    with gr.Row():
                        epochs_input = gr.Number(label="Épocas", value=10)
                        batch_size_input = gr.Number(label="Tamaño de Lote", value=8)
                        lr_input = gr.Number(label="Tasa de Aprendizaje", value=0.001)
                        save_every_input = gr.Number(label="Guardar Punto de Control Cada N Épocas", value=0)
                    device_configs[device] = [
                        epochs_input,
                        batch_size_input,
                        lr_input,
                        save_every_input
                    ]

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

        # Collect all inputs dynamically for each device
        def collect_inputs():
            inputs = [pretrain_dir_input, pretrain_output_dir_input]
            for device, widgets in device_configs.items():
                inputs.extend(widgets)
            return inputs

        pretrain_event = pretrain_btn.click(
            pretrain_batch_interface,
            inputs=collect_inputs(),
            outputs=pretrain_output,
        )
        pretrain_stop_btn.click(
            lambda: "Se envió la señal para detener el preentrenamiento. El proceso se detendrá si verifica la cancelación.",
            None,
            pretrain_output,
            cancels=[pretrain_event]
        )
    with gr.TabItem("Evaluar Backbone Preentrenado"):
        gr.Markdown("## Evaluar Backbone Preentrenado")
        gr.Markdown("Seleccione un archivo de backbone preentrenado (.pt) para ver su estructura y el número de parámetros.")
        with gr.Row():
            eval_backbone_input = gr.Textbox(
                label="Ruta al archivo backbone preentrenado (.pt)",
                placeholder="./pretrained_backbone.pt",
                info="Ruta al archivo .pt del backbone preentrenado"
            )
        eval_backbone_output = gr.Textbox(label="Resumen del Backbone", interactive=False, lines=10)
        eval_backbone_btn = gr.Button("Evaluar Backbone")
        eval_backbone_btn.click(
            eval_pretrained_backbone_interface,
            inputs=eval_backbone_input,
            outputs=eval_backbone_output
        )
