import spaces
import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
import torch
from scipy.io.wavfile import write
from diffusers import DiffusionPipeline
from transformers import pipeline
from pathlib import Path

load_dotenv()
hf_token = os.getenv("HF_TKN")

device_id = 0 if torch.cuda.is_available() else -1

captioning_pipeline = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=device_id
)

pipe = DiffusionPipeline.from_pretrained(
    "cvssp/audioldm2",
    use_auth_token=hf_token
)

@spaces.GPU(duration=120)
def analyze_image_with_free_model(image_file):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_file)
            temp_image_path = temp_file.name

        results = captioning_pipeline(temp_image_path)
        if not results or not isinstance(results, list):
            return "Error: Could not generate caption.", True
        
        caption = results[0].get("generated_text", "").strip()
        if not caption:
            return "No caption was generated.", True
        return caption, False

    except Exception as e:
        return f"Error analyzing image: {e}", True

@spaces.GPU(duration=120)
def get_audioldm_from_caption(caption):
    try:
        pipe.to("cuda")
        audio_output = pipe(
            prompt=caption,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        pipe.to("cpu")
        audio = audio_output.audios[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            write(temp_wav.name, 16000, audio)
            return temp_wav.name

    except Exception as e:
        print(f"Error generating audio from caption: {e}")
        return None

css = """
#col-container{
    margin: 0 auto;
    max-width: 800px;
    }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
    <h1 style="text-align: center;">🎶 Generate Sound Effects from Image</h1>
    <p style="text-align: center;">
        ⚡ Powered by <a href="https://bilsimaging.com" target="_blank">Bilsimaging</a>
    </p>
        """)

    gr.Markdown("""
    Welcome to this unique sound effect generator! This tool allows you to upload an image and generate a 
    descriptive caption and a corresponding sound effect, all using free, open-source models on Hugging Face.
    
    **💡 How it works:**
    1. **Upload an image**: Choose an image that you'd like to analyze.
    2. **Generate Description**: Click on 'Generate Description' to get a textual description of your uploaded image.
    3. **Generate Sound Effect**: Based on the image description, click on 'Generate Sound Effect' to create a 
       sound effect that matches the image context.
    
    Enjoy the journey from visual to auditory sensation with just a few clicks!
    """)

    image_upload = gr.File(label="Upload Image", type="binary")
    generate_description_button = gr.Button("Generate Description")
    caption_display = gr.Textbox(label="Image Description", interactive=False)
    generate_sound_button = gr.Button("Generate Sound Effect")
    audio_output = gr.Audio(label="Generated Sound Effect")

    gr.Markdown("""
    ## 👥 How You Can Contribute
    We welcome contributions and suggestions for improvements. Your feedback is invaluable 
    to the continuous enhancement of this application. 
    
    For support, questions, or to contribute, please contact us at 
    [contact@bilsimaging.com](mailto:contact@bilsimaging.com).
    
    Support our work and get involved by donating through 
    [Ko-fi](https://ko-fi.com/bilsimaging). - Bilel Aroua
    """)

    gr.Markdown("""
    ## 📢 Stay Connected
    This app is a testament to the creative possibilities that emerge when technology meets art. 
    Enjoy exploring the auditory landscape of your images!
    """)

    def update_caption(image_file):
        description, _ = analyze_image_with_free_model(image_file)
        return description

    def generate_sound(description):
        if not description or description.startswith("Error"):
            return None
        audio_path = get_audioldm_from_caption(description)
        return audio_path

    generate_description_button.click(
        fn=update_caption,
        inputs=image_upload,
        outputs=caption_display
    )

    generate_sound_button.click(
        fn=generate_sound,
        inputs=caption_display,
        outputs=audio_output
    )

demo.launch(debug=True, share=True)
