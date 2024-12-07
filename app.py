# Import necessary libraries
import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
import torch
from scipy.io.wavfile import write
from diffusers import DiffusionPipeline  
import google.generativeai as genai
from pathlib import Path

#there is an issue at this stage

# Load environment variables from .env file
load_dotenv()

#Google Generative AI for Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# Hugging Face token from environment variables
hf_token = os.getenv("HF_TKN")

def analyze_image_with_gemini(image_file):
    """
    Analyzes an uploaded image with Gemini and generates a descriptive caption.
    """
    try:
# Save uploaded image to a temporary file
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(image_file)

# Prepare the image data and prompt for Gemini
        image_parts = [{"mime_type": "image/jpeg", "data": Path(temp_image_path).read_bytes()}]
        prompt_parts = ["Describe precisely the image in one sentence.\n", image_parts[0], "\n"]
        generation_config = {"temperature": 0.05, "top_p": 1, "top_k": 26, "max_output_tokens": 4096}
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                           {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                           {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                           {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]
        model = genai.GenerativeModel(model_name="gemini-1.0-pro-vision-latest",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
        response = model.generate_content(prompt_parts)
        return response.text.strip(), False  # False indicates no error
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return "Error analyzing image with Gemini", True  # Indicates error with a message

def get_audioldm_from_caption(caption):
    """
    Generates sound from a caption using the AudioLDM-2 model.
    """
# Initialize the model
    pipe = DiffusionPipeline.from_pretrained("cvssp/audioldm2", use_auth_token=hf_token)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate audio from the caption
    audio_output = pipe(prompt=caption, num_inference_steps=50, guidance_scale=7.5)
    audio = audio_output.audios[0]  

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, 16000, audio)  

    return temp_file.name

# css
css="""
#col-container{
    margin: 0 auto;
    max-width: 800px;
    }

"""

# Gradio interface setup
with gr.Blocks(css=css) as demo:
    # Main Title and App Description
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
    <h1 style="text-align: center;">
           🎶 Generate Sound Effects from Image
        </h1>
         <p style="text-align: center;">
          ⚡  Powered by <a href="https://bilsimaging.com" _blank >Bilsimaging</a>
        </p>
    """)
    
    gr.Markdown("""
    Welcome to this unique sound effect generator! This tool allows you to upload an image and generate a descriptive caption and a corresponding sound effect. Whether you're exploring the sound of nature, urban environments, or anything in between, this app brings your images to auditory life.
    
    **💡 How it works:**
    1. **Upload an image**: Choose an image that you'd like to analyze.
    2. **Generate Description**: Click on 'Tap to Generate Description from the image' to get a textual description of your uploaded image.
    3. **Generate Sound Effect**: Based on the image description, click on 'Generate Sound Effect' to create a sound effect that matches the image context.
    
    Enjoy the journey from visual to auditory sensation with just a few clicks!
    
    For Example Demos sound effects generated , check out our [YouTube channel](https://www.youtube.com/playlist?list=PLwEbW4bdYBSDe6qAJRFiWGyHSW-JR-B0_)
    """)
    
# Interface Components
    image_upload = gr.File(label="Upload Image", type="binary")
    generate_description_button = gr.Button("Tap to Generate a Description from your image")
    caption_display = gr.Textbox(label="Image Description", interactive=False)  # Keep as read-only
    generate_sound_button = gr.Button("Generate Sound Effect")
    audio_output = gr.Audio(label="Generated Sound Effect")
# extra footer
    gr.Markdown("""## 👥 How You Can Contribute
        We welcome contributions and suggestions for improvements. Your feedback is invaluable to the continuous enhancement of this application. 
                
        For support, questions, or to contribute, please contact us at [contact@bilsimaging.com](mailto:contact@bilsimaging.com).
                
        Support our work and get involved by donating through [Ko-fi](https://ko-fi.com/bilsimaging). - Bilel Aroua
            """)
    gr.Markdown("""## 📢 Stay Connected
        this app is a testament to the creative possibilities that emerge when technology meets art. Enjoy exploring the auditory landscape of your images!
            """)
    # Function to update the caption display based on the uploaded image
    def update_caption(image_file):
        description, _ = analyze_image_with_gemini(image_file)
        return description

    # Function to generate sound from the description
    def generate_sound(description):
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



# Launch the Gradio app
demo.launch(debug=True, share=True)
