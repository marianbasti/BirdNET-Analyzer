import gradio as gr
from birdnet_analyzer.torch_gradio_pretrain import create_pretrain_tab
from birdnet_analyzer.torch_gradio_finetune import create_finetune_tab
from birdnet_analyzer.torch_gradio_eval import create_eval_tab
from birdnet_analyzer.torch_gradio_classify import create_classify_tab
from birdnet_analyzer.torch_gradio_vis import create_vis_tab

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador de Audio BirdNet (PyTorch)")
    with gr.Tabs():
        create_pretrain_tab()
        create_finetune_tab()
        create_eval_tab()
        create_classify_tab()
        create_vis_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
