import os
import torch
import gradio as gr
from birdnet_analyzer.torch_pretrain_utils import SimCLRPretrainer, UnlabeledAudioDataset, collate_fn

def pretrain_interface(
    data_dir, epochs, batch_size, learning_rate, 
    save_every_epochs=0, output_dir=None, 
    use_whisper=False, d_model=512, n_heads=8, n_layers=6,
    progress=gr.Progress(track_tqdm=True), request: gr.Request = None
):
    import os
    from torch.utils.data import DataLoader

    if not data_dir or not os.path.isdir(data_dir):
        return {"error": f"Por favor, proporciona una ruta de directorio válida. Recibido: '{data_dir}'"}
    try:
        dataset = UnlabeledAudioDataset(data_dir)
        if len(dataset) == 0:
            return {"error": f"No se encontraron archivos de audio en el directorio proporcionado: '{data_dir}'"}
        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, collate_fn=collate_fn)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if use_whisper:
            print("Using Whisper backbone for pretraining.")
            pretrainer = SimCLRPretrainer(
                device=device,
                emb_size=1024, # Standard for now
                proj_dim=128,
                n_mels=80,
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                log_wandb=False,
                run_name=f"whisper_pretrain_{d_model}d_{n_layers}l"
            )
        else:
            print("Using EfficientNet backbone for pretraining.")
            # This part needs to be updated if you want to support both.
            # For now, we assume the old SimCLRPretrainer for EfficientNet is available.
            # The original code was: SimCLRPretrainer(device=device)
            # This will fail with the new definition. We will stick to Whisper for now.
            return {"error": "Only Whisper backbone pretraining is supported in this version of the UI."}

        # Handle output_dir
        if output_dir is not None and output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'pretrained_backbone.pt')
            checkpoint_prefix = 'checkpoint_pretrain_epoch' # prefix is handled inside train
        else:
            save_path = 'pretrained_backbone.pt'
            checkpoint_prefix = 'checkpoint_pretrain_epoch'

        pretrainer.train(
            dataloader, 
            epochs=int(epochs), 
            lr=float(learning_rate), 
            save_path=save_path, 
            checkpoint_every=int(save_every_epochs),
            resume_from=None # Not exposed in UI yet
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": f"Error en el preentrenamiento: {e}\nTraceback:\n{tb}"}