import streamlit as st
import requests
import json
import time
import uuid
import os
import io
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass
from PIL import Image
import hashlib
# ============================================================================
# BLOCK 1: IMPORTING LIBRARIES AND SETTING UP THE PAGE
# ============================================================================
st.set_page_config(
    page_title="Venice AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.venice.ai',
        'Report a bug': 'https://venice.ai/support',
        'About': 'Venice AI Assistant Suite v6.0'
    }
)
# ============================================================================
# BLOCK 2: LOGGING SETUP AND API KEY CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# ============================================================================
# BLOCK 3: DEFINING CONSTANTS AND VENICESETTINGS DATACLASS
# ============================================================================
# Constants for magic numbers/strings
MAX_TEXT_CHARS = 4096
MAX_IMAGE_SEED = 2**32 - 1 # A more reasonable max seed value
DEFAULT_HISTORY_COUNT = 5
@dataclass
class VeniceSettings:
    """Venice AI API configuration"""
    base_url: str = "https://api.venice.ai/api/v1"
    chat_path: str = "/chat/completions"
    image_path: str = "/image/generate"
    audio_path: str = "/audio/speech"
    models_path: str = "/models"
    timeout: int = 60
    stream_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    @property
    def chat_endpoint(self) -> str: return f"{self.base_url}{self.chat_path}"
    @property
    def image_endpoint(self) -> str: return f"{self.base_url}{self.image_path}"
    @property
    def audio_endpoint(self) -> str: return f"{self.base_url}{self.audio_path}"
    @property
    def models_endpoint(self) -> str: return f"{self.base_url}{self.models_path}"
# Real Venice AI Models (from API specification) - Kept as constants for default/fallback
VENICE_MODELS = {
    'text': [
        "venice-uncensored", "qwen-2.5-qwq-32b", "qwen3-4b", "mistral-31-24b",
        "qwen3-235b", "llama-3.2-3b", "llama-3.3-70b", "llama-3.1-405b",
        "dolphin-2.9.2-qwen2-72b", "qwen-2.5-vl", "qwen-2.5-coder-32b",
        "deepseek-r1-671b", "deepseek-coder-v2-lite"
    ],
    'image': [
        "hidream", "flux-dev", "flux-dev-uncensored", "stable-diffusion-3.5",
        "venice-sd35", "flux.1-krea", "lustify-sdxl", "pony-realism",
        "wai-Illustrious"
    ],
    'audio': [
        "tts-kokoro"
    ]
}
# Venice AI Voice Options (from API specification)
VENICE_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "zf_xiaobei",
    "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "ff_siwis", "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha",
    "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora",
    "pm_alex", "pm_santa", "ef_dora", "em_alex", "em_santa"
]
# Image Style Presets (from API specification)
IMAGE_STYLES = [
    None, "3D Model", "Analog Film", "Anime", "Cinematic",
    "Comic Book", "Digital Art", "Fantasy", "Neon Punk", "Photographic"
]
# ============================================================================
# BLOCK 4: DEFINING THE MODERN UI STYLING
# ============================================================================
def inject_custom_css():
    """Inject optimized CSS for better performance and readability"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
 
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
    }
 
    /* Chat Messages - Enhanced Readability */
    .stChatMessage {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin: 12px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        color: #1f2937 !important;
    }
 
    /* Buttons - Modern gradient design */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
    }
 
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px rgba(37, 99, 235, 0.3) !important;
    }
 
    /* Image containers */
    .image-container {
        background: #f9fafb;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid #e5e7eb;
    }
 
    /* Success alerts */
    .success-alert {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 12px 0;
    }
 
    /* Error alerts */
    .error-alert {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 12px 0;
    }
 
    /* API Key display */
    .api-key-display {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 14px;
        color: #374151;
        word-break: break-all;
    }
 
    /* Loading spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
 
    /* Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%) !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
 
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%) !important;
    }
 
    /* File uploader */
    section[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #f9fafb 0%, #ffffff 100%) !important;
        border: 2px dashed #9ca3af !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
 
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6 !important;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    }
 
    /* Status indicators */
    .status-success {
        background: #dcfce7;
        color: #166534;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
    }
 
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
    }
 
    .status-pending {
        background: #fef3c7;
        color: #92400e;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
# ============================================================================
# BLOCK 5: DEFINING THE SESSIONSTATE CLASS
# ============================================================================
class SessionState:
    """Optimized session state management"""
 
    @staticmethod
    def initialize():
        """Initialize all session state variables with proper defaults"""
        defaults = {
            'messages': [],
            'conversation_id': str(uuid.uuid4()),
            'api_key': None, # Initialize as None, will be set by user/secrets
            'api_validated': False,
            'total_tokens': 0,
            'generated_images': [],
            'generated_audio': [],
            'current_model': None,
            'temperature': 0.7,
            'system_prompt': None,
            'is_streaming': False,
            'debug_mode': False,
            'error_log': [],
            'available_models': {'text': [], 'image': [], 'audio': []},
            'image_history': [],
            'current_mode': 'chat' # Default mode
        }
     
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        # Attempt to load API key from secrets or environment if not already set
        if st.session_state.api_key is None:
            if 'VENICE_API_KEY' in st.secrets:
                st.session_state.api_key = st.secrets['VENICE_API_KEY']
            elif os.getenv('VENICE_API_KEY'):
                st.session_state.api_key = os.getenv('VENICE_API_KEY')
    @staticmethod
    def reset_conversation():
        """Reset conversation state efficiently"""
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.total_tokens = 0
        st.rerun()
# ============================================================================
# BLOCK 6: VENICE AI CLIENT CLASS
# ============================================================================
class VeniceAIClient:
    """Production-ready Venice AI client with enhanced functionality"""
 
    def __init__(self, api_key: str, config: VeniceSettings):
        if not api_key:
            raise ValueError("API Key cannot be empty.")
        self.api_key = api_key
        self.settings = config
        self.session = self._create_session()
 
    def _create_session(self) -> requests.Session:
        """Create optimized requests session"""
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Venice-AI-Assistant/6.0',
            'Accept': 'application/json'
        })
        return session
 
    def validate_api_key(self) -> Tuple[bool, str, Dict[str, List[str]]]:
        """Enhanced API key validation with model fetching. Returns models, not updates session state."""
        try:
            response = self.session.get(
                self.settings.models_endpoint,
                timeout=10
            )
            if response.status_code == 200:
                models_data = response.json()
                available_models = {'text': [], 'image': [], 'audio': []}
                if 'data' in models_data:
                    available_models['text'] = [m['id'] for m in models_data['data'] if m['type'] == 'text']
                    available_models['image'] = [m['id'] for m in models_data['data'] if m['type'] == 'image']
                    available_models['audio'] = [m['id'] for m in models_data['data'] if m['type'] == 'tts']
              
                # Fallback to default models if API returns empty lists
                available_models['text'] = available_models['text'] or VENICE_MODELS['text']
                available_models['image'] = available_models['image'] or VENICE_MODELS['image']
                available_models['audio'] = available_models['audio'] or VENICE_MODELS['audio']
                return True, "API key validated successfully", available_models
            elif response.status_code == 401:
                return False, "Invalid API key", {'text': [], 'image': [], 'audio': []}
            else:
                return False, f"Validation failed: {response.status_code} - {response.text}", {'text': [], 'image': [], 'audio': []}
        except requests.exceptions.RequestException as e:
            logger.error(f"API key validation connection error: {str(e)}")
            return False, f"Connection error during validation: {str(e)}", VENICE_MODELS # Fallback with default models on connection error
        except Exception as e:
            logger.error(f"Unexpected error during API key validation: {str(e)}")
            return False, f"An unexpected error occurred: {str(e)}", VENICE_MODELS # Fallback with default models on unexpected error
 
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        stream: bool = True,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Optimized chat completion with better error handling"""
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': stream
        }
        if max_tokens:
            payload['max_tokens'] = max_tokens
     
        try:
            response = self.session.post(
                self.settings.chat_endpoint,
                json=payload,
                stream=stream,
                timeout=self.settings.stream_timeout
            )
         
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
         
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        if line_text.strip() == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line_text[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                                # Optionally, extract token usage if available in stream chunks
                                # if 'usage' in data and 'total_tokens' in data['usage']:
                                # st.session_state.total_tokens += data['usage']['total_tokens']
                        except json.JSONDecodeError:
                            logger.warning(f"JSONDecodeError in chat stream: {line_text}")
                            continue
                         
        except requests.exceptions.Timeout:
            logger.error("Chat completion request timed out.")
            yield "⏰ Error: Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            logger.error("Chat completion connection failed.")
            yield "🔌 Error: Connection failed. Please check your internet connection."
        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            yield f"❌ Error: {error_msg}"
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            yield f"❌ Error: {str(e)}"
 
    def generate_image(
        self,
        prompt: str,
        model: str = "hidream",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        style_preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced image generation with full parameter support"""
        payload = {
            'prompt': prompt,
            'model': model,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'format': 'png',
            'return_binary': False # Assuming API returns base64 string
        }
     
        if negative_prompt:
            payload['negative_prompt'] = negative_prompt
        if seed is not None:
            payload['seed'] = seed
        if style_preset:
            payload['style_preset'] = style_preset
     
        try:
            response = self.session.post(
                self.settings.image_endpoint,
                json=payload,
                timeout=self.settings.timeout
            )
         
            response.raise_for_status()
         
            data = response.json()
         
            if 'images' in data and data['images']:
                images = []
                for img_b64 in data['images']:
                    if img_b64.startswith('data:'):
                        images.append(img_b64)
                    else:
                        images.append(f"data:image/png;base64,{img_b64}")
             
                return {
                    'success': True,
                    'images': images,
                    'prompt': prompt,
                    'model': model,
                    'timestamp': datetime.now(),
                    'id': data.get('id', str(uuid.uuid4()))
                }
            else:
                raise ValueError("No images found in API response.")
             
        except requests.exceptions.RequestException as e:
            error_data = e.response.json() if e.response and e.response.content else {}
            error_msg = error_data.get('error', str(e))
            logger.error(f"Image generation API error: {error_msg}")
            return {
                'success': False,
                'error': f"API Error {e.response.status_code if e.response else 'N/A'}: {error_msg}",
                'prompt': prompt,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt,
                'timestamp': datetime.now()
            }
 
    def text_to_speech(
        self,
        text: str,
        model: str = "tts-kokoro",
        voice: str = "af_sky",
        speed: float = 1.0
    ) -> bytes:
        """Enhanced text-to-speech generation"""
        payload = {
            'model': model,
            'input': text,
            'voice': voice,
            'speed': speed,
            'response_format': 'mp3'
        }
     
        try:
            response = self.session.post(
                self.settings.audio_endpoint,
                json=payload,
                timeout=self.settings.timeout
            )
         
            response.raise_for_status()
         
            return response.content
         
        except requests.exceptions.RequestException as e:
            error_msg = f"API Error {e.response.status_code if e.response else 'N/A'}: {e.response.text if e.response else str(e)}"
            logger.error(f"TTS API error: {error_msg}")
            raise Exception(f"Audio generation failed: {error_msg}") from e
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            raise Exception(f"Audio generation failed: {str(e)}") from e
# ============================================================================
# BLOCK 7: ENHANCED IMAGE GENERATOR CLASS
# ============================================================================
class ImageGenerator:
    """Enhanced image generation interface with advanced controls"""
 
    def __init__(self, client: VeniceAIClient):
        self.client = client
 
    def render(self):
        """Render enhanced image generation interface"""
        st.markdown("### 🎨 Advanced Image Generation")
     
        with st.form("image_generation_form", clear_on_submit=False):
            col1, col2 = st.columns([3, 1])
         
            with col1:
                prompt = st.text_area(
                    "Image Prompt",
                    height=120,
                    placeholder="A serene Japanese garden with cherry blossoms, koi pond reflecting the sunset, ultra realistic, 8k, masterpiece",
                    help="Describe your image in detail. More specific prompts yield better results."
                )
             
                negative_prompt = st.text_area(
                    "Negative Prompt (Optional)",
                    height=80,
                    placeholder="blurry, low quality, distorted, watermark",
                    help="Describe what you DON'T want in the image"
                )
         
            with col2:
                available_models = st.session_state.available_models.get('image', VENICE_MODELS['image'])
                model = st.selectbox(
                    "Model",
                    available_models,
                    help="Different models have different artistic styles",
                    key="image_model_selector"
                )
             
                dimension_presets = {
                    "Square (1024x1024)": (1024, 1024),
                    "Portrait (832x1216)": (832, 1216),
                    "Landscape (1216x832)": (1216, 832),
                    "Custom": (1024, 1024)
                }
             
                dimension_choice = st.selectbox("Dimensions", list(dimension_presets.keys()))
             
                if dimension_choice == "Custom":
                    col_w, col_h = st.columns(2)
                    with col_w:
                        width = st.number_input("Width", min_value=512, max_value=1280, value=1024, step=64)
                    with col_h:
                        height = st.number_input("Height", min_value=512, max_value=1280, value=1024, step=64)
                else:
                    width, height = dimension_presets[dimension_choice]
         
            with st.expander("🔧 Advanced Settings", expanded=False):
                col_adv1, col_adv2, col_adv3 = st.columns(3)
             
                with col_adv1:
                    steps = st.slider("Steps", 10, 50, 20, help="More steps = higher quality but slower")
                    cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 7.5, 0.5, help="How closely to follow the prompt")
             
                with col_adv2:
                    use_seed = st.checkbox("Use Custom Seed")
                    seed = st.number_input("Seed", min_value=0, max_value=MAX_IMAGE_SEED, value=42) if use_seed else None
             
                with col_adv3:
                    style_preset = st.selectbox("Style Preset", IMAGE_STYLES)
         
            col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
            with col_gen1:
                generate_btn = st.form_submit_button(
                    "🎨 Generate Image",
                    type="primary",
                    disabled=not prompt,
                    use_container_width=True
                )
            with col_gen2:
                num_images = st.selectbox("Count", [1, 2, 3, 4], index=0)
            with col_gen3:
                save_to_history = st.checkbox("Save to History", value=True)
     
        if generate_btn and prompt:
            self._generate_images(
                prompt, negative_prompt, model, width, height,
                steps, cfg_scale, seed, style_preset, num_images, save_to_history
            )
     
        self._display_image_history()
 
    def _generate_images(self, prompt, negative_prompt, model, width, height,
                        steps, cfg_scale, seed, style_preset, num_images, save_to_history):
        """Generate images with progress tracking"""
     
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
     
        generated_images = []
     
        for i in range(num_images):
            progress_bar.progress((i) / num_images)
            status_text.text(f"🎨 Generating image {i+1}/{num_images}...")
         
            try:
                current_seed = seed + i if seed is not None else None # Use 'is not None' for clarity
                result = self.client.generate_image(
                    prompt=prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=current_seed,
                    negative_prompt=negative_prompt,
                    style_preset=style_preset
                )
             
                if result['success']:
                    generated_images.extend(result['images'])
                 
                    if save_to_history:
                        st.session_state.image_history.append({
                            'prompt': prompt,
                            'negative_prompt': negative_prompt,
                            'model': model,
                            'images': result['images'],
                            'timestamp': result['timestamp'],
                            'settings': {
                                'width': width, 'height': height,
                                'steps': steps, 'cfg_scale': cfg_scale,
                                'seed': current_seed, 'style_preset': style_preset
                            }
                        })
                else:
                    st.error(f"❌ Generation {i+1} failed: {result['error']}")
                 
            except Exception as e:
                st.error(f"❌ Generation {i+1} failed: {str(e)}")
     
        progress_bar.progress(1.0)
        status_text.text(f"✅ Generated {len(generated_images)} image(s)")
     
        if generated_images:
            with results_container:
                st.markdown("#### 🖼️ Generated Images")
             
                cols = st.columns(min(len(generated_images), 3))
                for idx, image_data in enumerate(generated_images):
                    col_idx = idx % len(cols)
                 
                    with cols[col_idx]:
                        st.image(image_data, caption=f"Image {idx+1}")
                     
                        if image_data.startswith('data:'):
                            image_bytes = base64.b64decode(image_data.split(',')[1])
                            st.download_button(
                                f"⬇️ Download {idx+1}",
                                data=image_bytes,
                                file_name=f"venice_ai_{int(time.time())}_{idx}.png",
                                mime="image/png",
                                key=f"download_{idx}_{time.time()}"
                            )
     
        progress_bar.empty()
        status_text.empty()
 
    def _display_image_history(self):
        """Display image generation history"""
        if st.session_state.image_history:
            st.markdown("---")
            st.markdown("#### 📚 Generation History")
         
            col_hist1, col_hist2 = st.columns([3, 1])
            with col_hist1:
                if st.button("🗑️ Clear History", key="clear_image_history_btn"):
                    st.session_state.image_history = []
                    st.rerun()
            with col_hist2:
                show_count_options = [3, 5, 10, "All"]
                show_count_index = show_count_options.index(DEFAULT_HISTORY_COUNT) if DEFAULT_HISTORY_COUNT in show_count_options else 0
                show_count = st.selectbox("Show", show_count_options, index=show_count_index, key="image_history_show_count")
         
            # Optimized slicing and reversal
            history_items_reversed = list(reversed(st.session_state.image_history))
            if show_count != "All":
                history_items_reversed = history_items_reversed[:show_count]
         
            for idx, item in enumerate(history_items_reversed):
                with st.expander(
                    f"🎨 {item['prompt'][:60]}{'...' if len(item['prompt']) > 60 else ''}" +
                    f" | {item['model']} | {item['timestamp'].strftime('%H:%M:%S')}",
                    expanded=False
                ):
                 
                    if item['images']:
                        cols = st.columns(min(len(item['images']), 4))
                        for img_idx, img_data in enumerate(item['images']):
                            with cols[img_idx % len(cols)]:
                                st.image(img_data, use_column_width=True)
                             
                                if img_data.startswith('data:'):
                                    image_bytes = base64.b64decode(img_data.split(',')[1])
                                    st.download_button(
                                        "⬇️",
                                        data=image_bytes,
                                        file_name=f"venice_history_{idx}_{img_idx}.png",
                                        mime="image/png",
                                        key=f"hist_download_{idx}_{img_idx}_{item['timestamp'].timestamp()}"
                                    )
                 
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.text(f"Model: {item['model']}")
                        st.text(f"Dimensions: {item['settings']['width']}x{item['settings']['height']}")
                        st.text(f"Steps: {item['settings']['steps']}")
                    with col_info2:
                        st.text(f"CFG Scale: {item['settings']['cfg_scale']}")
                        st.text(f"Seed: {item['settings']['seed']}")
                        st.text(f"Style: {item['settings']['style_preset'] or 'None'}")
                 
                    if item.get('negative_prompt'):
                        st.text(f"Negative: {item['negative_prompt']}")
# ============================================================================
# BLOCK 8: DEFINING THE CHATINTERFACE CLASS
# ============================================================================
class ChatInterface:
    """Enhanced chat interface"""
 
    def __init__(self, client: VeniceAIClient):
        self.client = client
 
    def render(self):
        """Render optimized chat interface"""
        col1, col2, col3, col4 = st.columns([6, 2, 2, 2])
     
        with col1:
            st.markdown("### 💬 AI Chat Assistant")
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        with col3:
            st.metric("Tokens", f"{st.session_state.total_tokens:,}")
        with col4:
            if st.button("🔄 Clear Chat", use_container_width=True, key="clear_chat_btn"):
                SessionState.reset_conversation()
     
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
     
        if prompt := st.chat_input("Type your message...", disabled=st.session_state.is_streaming):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
         
            with st.chat_message('user'):
                st.markdown(prompt)
         
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                full_response = ""
             
                st.session_state.is_streaming = True
             
                try:
                    api_messages = st.session_state.messages.copy()
                 
                    for chunk in self.client.chat_completion(
                        messages=api_messages,
                        model=st.session_state.current_model,
                        temperature=st.session_state.temperature
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                 
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
                    # Simple word count as a heuristic if API doesn't provide token count
                    st.session_state.total_tokens += len(full_response.split()) + len(prompt.split())
                 
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
             
                finally:
                    st.session_state.is_streaming = False
# ============================================================================
# BLOCK 9: DEFINING THE AUDIOINTERFACE CLASS
# ============================================================================
class AudioInterface:
    """Enhanced audio generation interface"""
 
    def __init__(self, client: VeniceAIClient):
        self.client = client
 
    def render(self):
        """Render TTS interface with Venice voices"""
        st.markdown("### 🎤 Text-to-Speech Generation")
     
        with st.form("tts_form"):
            col1, col2 = st.columns([3, 1])
         
            with col1:
                text = st.text_area(
                    "Text to Convert",
                    height=150,
                    max_chars=MAX_TEXT_CHARS,
                    placeholder="Enter the text you want to convert to speech..."
                )
                char_count = len(text) if text else 0
                st.caption(f"Characters: {char_count}/{MAX_TEXT_CHARS}")
         
            with col2:
                voice = st.selectbox("Voice", VENICE_VOICES)
                speed = st.slider("Speed", 0.25, 4.0, 1.0, 0.25)
                # BUG FIX: Use current_model from session state
                model = st.session_state.current_model or VENICE_MODELS['audio'][0] # Fallback if not set
         
            generate_audio = st.form_submit_button(
                "🎤 Generate Audio",
                type="primary",
                disabled=not text
            )
     
        if generate_audio and text:
            with st.spinner("🎵 Generating audio..."):
                try:
                    audio_data = self.client.text_to_speech(
                        text=text,
                        model=model,
                        voice=voice,
                        speed=speed
                    )
                 
                    st.audio(audio_data, format='audio/mp3')
                 
                    st.download_button(
                        "⬇️ Download Audio",
                        data=audio_data,
                        file_name=f"venice_tts_{int(time.time())}.mp3",
                        mime="audio/mp3"
                    )
                 
                    st.success("✅ Audio generated successfully!")
                 
                    st.session_state.generated_audio.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'voice': voice,
                        'speed': speed,
                        'timestamp': datetime.now(),
                        'audio_data': audio_data # Store actual audio data for history playback
                    })
                 
                except Exception as e:
                    st.error(f"❌ Audio generation failed: {str(e)}")
     
        if st.session_state.generated_audio:
            st.markdown("---")
            st.markdown("#### 🎵 Audio History")
         
            # Use DEFAULT_HISTORY_COUNT constant
            for idx, audio_item in enumerate(reversed(st.session_state.generated_audio[-DEFAULT_HISTORY_COUNT:])):
                with st.expander(
                    f"🎤 {audio_item['text']} | {audio_item['voice']} | {audio_item['timestamp'].strftime('%H:%M:%S')}"
                ):
                    st.audio(audio_item['audio_data'], format='audio/mp3')
                    st.download_button(
                        "⬇️ Download",
                        data=audio_item['audio_data'],
                        file_name=f"venice_history_{idx}_{audio_item['timestamp'].timestamp()}.mp3",
                        mime="audio/mp3",
                        key=f"audio_download_{idx}_{audio_item['timestamp'].timestamp()}"
                    )
# ============================================================================
# BLOCK 10: CREATE OPTIMIZED SIDEBAR UI FUNCTION
# ============================================================================
def create_sidebar_ui() -> str:
    """Create optimized sidebar with proper model handling"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🚀 Venice AI</h1>
            <p style='color: #6b7280;'>Assistant Suite v6.0</p>
        </div>
        """, unsafe_allow_html=True)
     
        st.markdown("---")
     
        st.markdown("### 🔐 API Configuration")
     
        current_key = st.session_state.api_key
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if current_key and len(current_key) > 12 else (current_key or "Not Set")
     
        st.markdown("**Current API Key:**")
        st.markdown(f'<div class="api-key-display">{masked_key}</div>', unsafe_allow_html=True)
     
        with st.expander("🔧 Change API Key", expanded=False):
            new_api_key = st.text_input(
                "Enter new API key",
                type="password",
                placeholder="Enter your Venice AI API key here...",
                key="new_api_key_input"
            )
         
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Key", type="secondary", key="update_api_key_btn"):
                    if new_api_key and new_api_key.strip():
                        st.session_state.api_key = new_api_key.strip()
                        st.session_state.api_validated = False
                        st.success("API key updated! Please validate.")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
         
            with col2:
                if st.button("Clear Key", key="clear_api_key_btn"):
                    st.session_state.api_key = None
                    st.session_state.api_validated = False
                    st.success("API key cleared!")
                    st.rerun()
     
        if not st.session_state.api_validated or not st.session_state.api_key:
            st.warning("Please validate your API key to proceed.")
            if st.button("✅ Validate API Key", type="primary", use_container_width=True, key="validate_api_key_btn"):
                if st.session_state.api_key:
                    with st.spinner("Validating..."):
                        client = VeniceAIClient(st.session_state.api_key, VeniceSettings())
                        valid, message, available_models = client.validate_api_key()
                     
                        if valid:
                            st.session_state.api_validated = True
                            st.session_state.available_models = available_models
                            st.success(message)
                            st.rerun()
                        else:
                            st.session_state.api_validated = False
                            st.error(message)
                else:
                    st.error("No API key entered. Please enter or update your key.")
            st.markdown("[Get API Key](https://venice.ai/api-keys)")
            st.stop() # Stop further rendering until key is validated
        st.markdown("### 🎯 Mode Selection")
        mode_options = {"💬 Chat": "chat", "🎨 Images": "images", "🎤 Audio": "audio"}
        selected_mode_label = st.radio(
            "Choose Mode",
            list(mode_options.keys()),
            label_visibility="collapsed",
            key="mode_selector",
            index=list(mode_options.keys()).index(next(key for key, value in mode_options.items() if value == st.session_state.current_mode))
        )
        st.session_state.current_mode = mode_options[selected_mode_label]
     
        st.markdown("### 🤖 Model Settings")
     
        if st.session_state.current_mode == "chat":
            models = st.session_state.available_models.get('text', VENICE_MODELS['text'])
            current_model = st.selectbox("Chat Model", models, key="chat_model_selector")
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            st.session_state.current_model = current_model
            st.session_state.temperature = temperature
         
        elif st.session_state.current_mode == "images":
            models = st.session_state.available_models.get('image', VENICE_MODELS['image'])
            current_model = st.selectbox("Image Model", models, key="image_model_selector_sidebar") # Unique key
            st.session_state.current_model = current_model
         
        else: # Audio
            models = st.session_state.available_models.get('audio', VENICE_MODELS['audio'])
            current_model = st.selectbox("TTS Model", models, key="audio_model_selector")
            st.session_state.current_model = current_model
     
        st.markdown("---")
     
        st.markdown("### ⚡ Quick Actions")
     
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Conversation", use_container_width=True, key="reset_conv_btn"):
                SessionState.reset_conversation()
     
        with col2:
            export_data = {
                'messages': st.session_state.messages,
                'image_history': st.session_state.image_history,
                'audio_history_summary': [{'text': item['text'], 'voice': item['voice'], 'timestamp': item['timestamp']} for item in st.session_state.generated_audio],
                'timestamp': datetime.now().isoformat(),
                'model_used': st.session_state.current_model,
                'api_key_used': f"{st.session_state.api_key[:8]}...{st.session_state.api_key[-4:]}" if st.session_state.api_key else "N/A"
            }
         
            st.download_button(
                "📊 Export Data",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"venice_export_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
                key="export_data_btn"
            )
     
        st.markdown("---")
        st.markdown("### 🔑 API Status")
     
        key_status_text = "✅ Validated" if st.session_state.api_validated else "❌ Not Validated"
        key_status_class = "status-success" if st.session_state.api_validated else "status-error"
     
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="{key_status_class}">{key_status_text}</div>', unsafe_allow_html=True)
        with col2:
            st.text(f"Key: {masked_key}")
     
        if st.session_state.messages or st.session_state.image_history or st.session_state.generated_audio:
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
         
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chat Messages", len(st.session_state.messages))
                st.metric("Images Generated", len(st.session_state.image_history))
            with col2:
                st.metric("Total Tokens", f"{st.session_state.total_tokens:,.0f}")
                st.metric("Audio Files", len(st.session_state.generated_audio))
     
        debug_mode = st.checkbox("🔧 Debug Mode", key="debug_mode_checkbox")
        st.session_state.debug_mode = debug_mode # Update session state for main app logic
        if debug_mode:
            with st.expander("Debug Info", expanded=False):
                st.json({
                    'session_id': st.session_state.conversation_id,
                    'api_validated': st.session_state.api_validated,
                    'current_mode': st.session_state.current_mode,
                    'current_model': st.session_state.current_model,
                    'api_key_length': len(st.session_state.api_key) if st.session_state.api_key else 0,
                    'available_models': {k: len(v) for k, v in st.session_state.available_models.items()},
                    'messages_count': len(st.session_state.messages),
                    'image_history_count': len(st.session_state.image_history),
                    'audio_history_count': len(st.session_state.generated_audio)
                })
     
        return st.session_state.current_mode
# ============================================================================
# BLOCK 11: DEFINING THE MAIN FUNCTION
# ============================================================================
def main():
    """Main application with optimized performance"""
    inject_custom_css()
    SessionState.initialize()
 
    # Get mode from sidebar (and handle API key validation)
    mode = create_sidebar_ui()
 
    # Client creation after API key is validated and available
    client = VeniceAIClient(st.session_state.api_key, VeniceSettings())
 
    st.markdown("# 🚀 Venice AI Assistant Suite")
    st.markdown("*Advanced AI Chat, Image Generation, and Text-to-Speech Platform*")
 
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        key_status = "✅ Validated" if st.session_state.api_validated else "⏳ Pending Validation"
        st.markdown(f"**API Status:** {key_status}")
    with col2:
        masked_key = f"{st.session_state.api_key[:8]}...{st.session_state.api_key[-4:]}" if st.session_state.api_key and len(st.session_state.api_key) > 12 else (st.session_state.api_key or "Not Set")
        st.markdown(f"**Key:** `{masked_key}`")
    with col3:
        current_model = st.session_state.current_model or "Not Selected"
        st.markdown(f"**Model:** {current_model}")
    with col4:
        if st.button("🔄 Refresh", key="main_refresh_btn"):
            st.rerun()
 
    st.markdown("---")
 
    try:
        if mode == "chat":
            chat_interface = ChatInterface(client)
            chat_interface.render()
         
        elif mode == "images":
            image_generator = ImageGenerator(client)
            image_generator.render()
         
        elif mode == "audio":
            audio_interface = AudioInterface(client)
            audio_interface.render()
         
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
 
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #6b7280; font-size: 14px;'>"
        f"Venice AI Assistant v6.0 | Session: {st.session_state.conversation_id[:8]} | "
        f"Model: {st.session_state.current_model or 'Not Selected'} | "
        f"API Key: {masked_key} | "
        f"Powered by Venice AI"
        f"</div>",
        unsafe_allow_html=True
    )
# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
 