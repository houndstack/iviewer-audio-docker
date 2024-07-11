import gradio as gr
from transformers import pipeline
import numpy as np
import torch
import time



transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    sample = {'array': y, 'sampling_rate': sr}
    return pipe(sample.copy(), batch_size=8, return_timestamps=True, generate_kwargs={"task": "translate"})["chunks"]
    #return transcriber({"sampling_rate": sr, "raw": y})["text"]

def starttimer(request: gr.Request):
    for att in dir(request):
        print (att, getattr(request,att))
    
    if request:
        print("Request headers dictionary:", request.headers)
        print("IP address:", request.client.host)
        print("Query parameters:", dict(request.query_params))
        print("Path parameters:", request.path_params)
    print(round(time.time() * 1000))

# demo = gr.Interface(
#     transcribe,
#     gr.Audio(sources=["microphone"]),
#     "text",
    
# )
with gr.Blocks() as demo:
    inp = gr.Audio(source='microphone')
    out = gr.Textbox()
    timeout = gr.Textbox()
    inp.start_recording(starttimer)
    inp.change(transcribe, inp, out)
    

demo.launch()
