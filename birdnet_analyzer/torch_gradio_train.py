import os
import torch
import gradio as gr
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from birdnet_analyzer.torch_train_utils import AudioDataset, train_model, evaluate_model
from birdnet_analyzer.torch_model import BirdNetTorchModel
from birdnet_analyzer.whisper_utils import load_pretrained_whisper_backbone

def pad_or_truncate(waveform, target_length=2880000):
    length = waveform.shape[-1]
    if length == target_length:
        return waveform
    elif length > target_length:
        return waveform[..., :target_length]
    else:
        pad_width = target_length - length
        return torch.nn.functional.pad(waveform, (0, pad_width))

# Add request: gr.Request to function signature
def train_interface(data_dir, model_path, epochs, batch_size, learning_rate, output_dir=None, progress=gr.Progress(track_tqdm=True), request: gr.Request = None): # Added request
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from birdnet_analyzer.torch_train_utils import AudioDataset, train_model
    from birdnet_analyzer.whisper_utils import load_pretrained_whisper_backbone
    import torchaudio
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model
    try:
        # Try to load as a whisper backbone if model_path is provided
        if model_path is not None:
            try:
                # Use the new utility to load a model with a pretrained whisper backbone
                model = load_pretrained_whisper_backbone(
                    checkpoint_path=model_path.name,
                    num_classes=len(class_names),
                    device=device,
                    freeze_backbone=False # Example: unfreeze for fine-tuning
                )
                print("Loaded model with pretrained Whisper backbone.")
            except Exception as e:
                 return {"error": f"Error loading Whisper backbone: {e}"}
        else:
            # Fallback to creating a new model (you might want to specify architecture here)
            model = BirdNetTorchModel(num_classes=len(class_names))

        model = model.to(device)

    except Exception as e:
        return {"error": f"Error al cargar el modelo: {e}"}
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
    import numpy as np
    from sklearn.model_selection import train_test_split
    idxs = np.arange(len(audio_data))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=42)
    train_audio, val_audio = audio_data[train_idx], audio_data[val_idx]
    train_labels, val_labels = label_data[train_idx], label_data[val_idx]
    train_ds = AudioDataset(train_audio, train_labels)
    val_ds = AudioDataset(val_audio, val_labels)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size))
    # Train model
    from birdnet_analyzer.torch_train_utils import evaluate_model
    # Handle output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        checkpoint_prefix = os.path.join(output_dir, 'checkpoint_finetune_epoch')
    else:
        best_model_path = 'best_model.pt'
        checkpoint_prefix = 'checkpoint_finetune_epoch'
    try:
        # NOTE: train_model would need to be modified to accept a stop_callback
        # e.g., stop_callback=lambda: request.is_cancelled if request else False
        train_model(model, train_loader, val_loader, epochs=int(epochs), lr=float(learning_rate), device=device, progress=progress, best_model_path=best_model_path, checkpoint_prefix=checkpoint_prefix)
    except TypeError: # Fallback for older train_model without checkpointing args
        train_model(model, train_loader, val_loader, epochs=int(epochs), lr=float(learning_rate), device=device, progress=progress)



    # After training, evaluate on val set
    val_loss, val_metrics = evaluate_model(model, val_loader, device, return_metrics=True)
    results = {"val_loss": val_loss, "val_metrics": val_metrics, "class_names": class_names, "best_model_path": best_model_path}
    return results