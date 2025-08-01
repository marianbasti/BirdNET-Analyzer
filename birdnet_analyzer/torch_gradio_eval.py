import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from birdnet_analyzer.torch_train_utils import AudioDataset, evaluate_model
from birdnet_analyzer.torch_model import BirdNetTorchModel

# Pad or truncate waveform to fixed length (1 minute = 2,880,000 samples at 48kHz)
def pad_or_truncate(waveform, target_length=2880000):
    length = waveform.shape[-1]
    if length == target_length:
        return waveform
    elif length > target_length:
        return waveform[..., :target_length]
    else:
        pad_width = target_length - length
        return torch.nn.functional.pad(waveform, (0, pad_width))

# --- Evaluation Interface ---
def eval_interface(data_dir, model_path):
    import os
    import numpy as np
    import torch
    import torchaudio
    # Scan subdirectories for classes
    if not os.path.isdir(data_dir):
        return {"error": "Por favor, proporciona una ruta de directorio válida."}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        return {"error": "No se encontraron subdirectorios de clase en el directorio proporcionado."}
    # Load model
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = BirdNetTorchModel(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
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
    # Evaluate model with metrics
    from torch.utils.data import DataLoader, Dataset
    from birdnet_analyzer.torch_train_utils import AudioDataset, evaluate_model
    val_ds = AudioDataset(audio_data, label_data)
    val_loader = DataLoader(val_ds, batch_size=8)
    val_loss, val_metrics = evaluate_model(model, val_loader, device, return_metrics=True)
    results = {"val_loss": val_loss, "val_metrics": val_metrics, "class_names": class_names}
    return results