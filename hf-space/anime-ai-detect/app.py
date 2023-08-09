import gradio
from transformers import pipeline

detection_pipeline = pipeline("image-classification", "saltacc/anime-ai-detect")


def detect(img):
    output = detection_pipeline(img, top_k=2)
    final = {}
    for d in output:
        final[d["label"]] = d["score"]
    return final


iface = gradio.Interface(fn=detect, inputs=gradio.Image(type="pil"), outputs=gradio.Label(label="result"))
iface.launch(server_name="0.0.0.0" , port=7865)
