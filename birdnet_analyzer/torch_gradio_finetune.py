import gradio as gr
import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from birdnet_analyzer.torch_model import BirdNetTorchModel
from birdnet_analyzer.torch_train_utils import AudioDataset, train_model, evaluate_model

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

def list_classes(data_dir):
    """Utility function to list classes in a directory"""
    if not os.path.isdir(data_dir):
        return ""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return ""
    return ", ".join(class_names)

def train_interface(data_dir, model_path, epochs, batch_size, learning_rate, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None):
    """Fine-tuning interface for supervised learning."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    
    # Load model
    try:
        model = BirdNetTorchModel(num_classes=len(class_names))
        if model_path and model_path.name:
            try:
                state = torch.load(model_path.name, map_location=device)
                # Try to load as full model first
                try:
                    model.load_state_dict(state)
                    print("Se cargó el modelo completo.")
                except Exception:
                    # If failed, try loading as backbone only by filtering keys
                    backbone_state = {k.replace('backbone.', ''): v for k, v in state.items() if k.startswith('backbone.')}
                    if not backbone_state:  # Try loading a raw backbone state dict
                        backbone_state = state
                    
                    model.backbone.load_state_dict(backbone_state, strict=False)
                    print("Se cargaron los pesos del backbone para el fine-tuning.")

            except Exception as e:
                return {"error": f"Error al cargar el modelo: {e}"}
        model = model.to(device)
    except Exception as e:
        return {"error": f"Error al inicializar el modelo: {e}"}
    
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
    idxs = np.arange(len(audio_data))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=42)
    train_audio, val_audio = audio_data[train_idx], audio_data[val_idx]
    train_labels, val_labels = label_data[train_idx], label_data[val_idx]
    
    train_ds = AudioDataset(train_audio, train_labels)
    val_ds = AudioDataset(val_audio, val_labels)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size))
    
    # Handle output_dir
    if output_dir is not None and output_dir.strip():
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        checkpoint_prefix = os.path.join(output_dir, 'checkpoint_finetune_epoch')
    else:
        best_model_path = 'best_model.pt'
        checkpoint_prefix = 'checkpoint_finetune_epoch'
    
    try:
        train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=int(epochs), 
            lr=float(learning_rate), 
            device=device, 
            progress=progress, 
            best_model_path=best_model_path, 
            checkpoint_prefix=checkpoint_prefix,
            use_focal_loss=False,  # Can be made configurable
            early_stopping_patience=10,
            scheduler_type='ReduceLROnPlateau'
        )
    except Exception as e:
        return {"error": f"Error durante el entrenamiento: {e}"}

    # After training, evaluate on val set
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_loss, val_metrics = evaluate_model(model, val_loader, device, return_metrics=True)
    
    # Save training parameters
    if output_dir and output_dir.strip():
        from birdnet_analyzer.utils import save_model_params
        params_dict = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_classes": len(class_names),
            "device": device,
            "val_loss": val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_f1": val_metrics['f1']
        }
        save_model_params(os.path.join(output_dir, 'model_params.csv'), params_dict)
    
    results = {
        "val_loss": val_loss, 
        "val_metrics": val_metrics, 
        "class_names": class_names, 
        "best_model_path": best_model_path
    }
    return results

def create_finetune_tab():
    """Create the fine-tuning tab interface."""
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
        )
        train_stop_btn.click(
            lambda: "Se envió la señal para detener el entrenamiento. El proceso se detendrá si verifica la cancelación.",
            None,
            train_output,
            cancels=[train_event]
        )
