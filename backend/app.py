import time
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)
import os
import sys
import uuid
import logging
import io
from typing import List, Tuple, Optional
import torch
import torchaudio
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import AutoModelForCausalLM
from yarngpt.audiotokenizer import AudioTokenizer
import numpy as np
from pydub import AudioSegment
import io
# Import the configuration
from config import Config

import subprocess
import sys

def split_text_into_chunks(text, max_length=200):
    """Split text into smaller chunks at sentence boundaries."""
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def combine_audio_files(audio_segments, crossfade_duration=100):
    """Combine multiple audio segments with smooth crossfade."""
    if not audio_segments:
        return None
    
    combined = audio_segments[0]
    for segment in audio_segments[1:]:
        combined = combined.append(segment, crossfade=crossfade_duration)
    
    return combined

def generate_audio_for_chunk(chunk, voice, tts_optimizer):
    """Generate audio for a single chunk of text."""
    with torch.inference_mode():
        prompt = tts_optimizer.audio_tokenizer.create_prompt(chunk, voice)
        input_ids = tts_optimizer.audio_tokenizer.tokenize_prompt(prompt)
        attention_mask = torch.ones_like(input_ids)
        
        output = tts_optimizer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            max_length=4000,
            pad_token_id=tts_optimizer.model.config.eos_token_id,
            eos_token_id=tts_optimizer.model.config.eos_token_id,
            no_repeat_ngram_size=3,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        
        codes = tts_optimizer.audio_tokenizer.get_codes(output)
        return tts_optimizer.audio_tokenizer.get_audio(codes)
    

# Configure logging based on Config
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """
    Check if FFmpeg is installed and accessible
    """
    try:
        # Try to run ffmpeg and capture its version
        result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        
        # Use .splitlines() to handle newline characters safely
        version_line = result.stdout.splitlines()[0]
        logger.info(f"FFmpeg version: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("FFmpeg not found. Audio processing may fail.")
        return False
    

class TTSOptimizer:
    def __init__(self, model, audio_tokenizer):
        """
        Initialize TTS Optimizer with advanced caching and optimization strategies
        
        Args:
            model: Pretrained language model
            audio_tokenizer: Audio tokenization utility
        """
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        
        # Advanced caching with LRU and additional memory management
        from functools import lru_cache
        self.generate_speech_cached = lru_cache(maxsize=100)(self._generate_speech_impl)
        
        # Performance tracking
        self.generation_times = []
        self.total_generations = 0

    def _generate_speech_impl(self, text: str, voice: str, 
                       temperature: float = 0.1, 
                       max_length: int = 4000) -> bytes:
        """
        Core speech generation implementation with advanced parameters
        """
        try:
            # Performance tracking with fallback
            start_time = time.time()
    
            # Create prompt and tokenize
            prompt = self.audio_tokenizer.create_prompt(text, voice)
            input_ids = self.audio_tokenizer.tokenize_prompt(prompt)

            # Ensure input is on the correct device and type
            input_ids = input_ids.to(self.model.device)

            with torch.inference_mode():
                # Advanced generation with multiple sampling strategies
                output = self.model.generate(
                    input_ids=input_ids,
                    temperature=temperature,
                    repetition_penalty=1.1,
                    max_length=max_length,
                    do_sample=True,
                    num_return_sequences=1,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.model.config.eos_token_id,
                    attention_mask=torch.ones_like(input_ids)
                )

            # Convert codes to audio
            codes = self.audio_tokenizer.get_codes(output)
            audio = self.audio_tokenizer.get_audio(codes)

            # Performance tracking
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            self.total_generations += 1

            # Convert to bytes for caching
            audio_buffer = io.BytesIO()
            torchaudio.save(audio_buffer, audio, sample_rate=24000, format='wav')
            return audio_buffer.getvalue()

        except Exception as e:
            logger.error(f"Speech generation error: {e}", exc_info=True)
            raise

    def batch_generate(self, texts: List[Tuple[str, str]], 
                       max_workers: Optional[int] = None) -> List[Optional[bytes]]:
        """
        Batch generation of multiple audio files with optional parallel processing
        
        Args:
            texts: List of (text, voice) tuples
            max_workers: Maximum number of parallel workers
        
        Returns:
            List of audio byte streams or None if generation fails
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit generation tasks
            future_to_index = {
                executor.submit(self.generate_speech_cached, text, voice): idx 
                for idx, (text, voice) in enumerate(texts)
            }

            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Batch generation failed for index {idx}: {e}")
        
        return results

    def get_performance_stats(self) -> dict:
        """
        Retrieve performance statistics for speech generation
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.generation_times:
            return {
                "total_generations": 0,
                "avg_generation_time": None,
                "min_generation_time": None,
                "max_generation_time": None
            }
        
        return {
            "total_generations": self.total_generations,
            "avg_generation_time": np.mean(self.generation_times),
            "min_generation_time": np.min(self.generation_times),
            "max_generation_time": np.max(self.generation_times)
        }

    def optimize_model(self):
        """
        Apply model optimization techniques
        """
        # Just put model in eval mode
        self.model.eval()
def create_app(model, audio_tokenizer):
    # Create TTS Optimizer
    tts_optimizer = TTSOptimizer(model, audio_tokenizer)
    tts_optimizer.optimize_model()

    # Flask Application Setup
    app = Flask(__name__)
    
    # Configure secret key
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    
    
    # Advanced CORS Configuration
    CORS(app, resources={
      r"/*": {
        "origins": [
                "http://localhost:5000", 
                "http://127.0.0.1:5000", 
                "http://localhost:3000", 
                "http://127.0.0.1:3000",
                "http://192.168.1.109:5000",  # Add your local IP
                "*"  # Use cautiously in development
            ],
            "methods": ["POST", "GET", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })

    # Rate Limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[Config.RATE_LIMIT_DEFAULT],
        storage_uri="memory://"
    )


    @app.route('/health', methods=['GET'])
    @limiter.limit("10 per minute")
    def health_check():
        """Health check endpoint with performance insights"""
        performance_stats = tts_optimizer.get_performance_stats()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "performance": performance_stats
        })

    @app.route('/generate', methods=['POST'])
    @limiter.limit("20 per hour")
    def generate_speech():
        try:
            data = request.json
            text = data.get('text', '').strip()
            voice = data.get('voice', 'idera')

            if not text:
                logger.error("No text provided")
                return jsonify({"error": "No text provided"}), 400

            logger.info(f"Generating speech for text: {text[:50]}... with voice: {voice}")

            def generate():
                try:
                    # Split text into chunks if it's long
                    chunks = split_text_into_chunks(text) if len(text) > 200 else [text]
                    audio_segments = []

                    # Generate audio for each chunk
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                        audio = generate_audio_for_chunk(chunk, voice, tts_optimizer)
                    
                        # Convert torch tensor to AudioSegment
                        buffer = io.BytesIO()
                        torchaudio.save(buffer, audio, sample_rate=24000, format='wav')
                        buffer.seek(0)
                        audio_segment = AudioSegment.from_wav(buffer)
                        audio_segments.append(audio_segment)

                    # Combine all segments with crossfade
                    final_audio = combine_audio_files(audio_segments)

                    # Save combined audio
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    temp_dir = os.path.join(current_dir, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f'audio_{os.urandom(8).hex()}.wav')
                
                    final_audio.export(temp_path, format='wav')
                    return temp_path

                except Exception as e:
                    logger.error(f"Generation error: {str(e)}", exc_info=True)
                    raise

            future = executor.submit(generate)
            try:
                temp_path = future.result(timeout=12000)  # 5 minutes timeout for long texts
                return send_file(temp_path, mimetype='audio/wav', as_attachment=True)
            except Exception as e:
                logger.error(f"Error in generation: {str(e)}", exc_info=True)
                return jsonify({
                    "error": "Audio generation failed",
                    "details": str(e)
                }), 500

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Audio generation failed",
                "details": str(e)
            }), 500
        

    # Optional: Batch generation endpoint
    @app.route('/batch-generate', methods=['POST'])
    @limiter.limit("10 per hour")
    def batch_generate_speech():
        """
        Batch generation of multiple audio files
        """
        try:
            data = request.json
            batch_inputs = data.get('inputs', [])

            # Validate batch inputs
            if not batch_inputs or len(batch_inputs) > 10:
                return jsonify({"error": "Invalid batch size"}), 400

            # Generate batch audio
            results = tts_optimizer.batch_generate(
                [(item['text'], item.get('voice', 'idera')) for item in batch_inputs]
            )

            # Prepare response
            response_data = []
            for idx, audio_bytes in enumerate(results):
                if audio_bytes:
                    response_data.append({
                        "index": idx,
                        "audio": audio_bytes.decode('latin-1'),  # Convert to base64 or similar if needed
                        "filename": f"batch_tts_{idx}.wav"
                    })

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return jsonify({"error": "Failed to generate batch speech"}), 500

    # Error handlers
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": "Rate limit exceeded"}), 429

    return app

def initialize_model():
    """
    Initialize TTS model and tokenizer with error handling
    """
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, 'downloaded_models')

        hf_path = "saheedniyi/YarnGPT"
        wav_tokenizer_config_path = os.path.join(MODEL_DIR, 'wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml')
        wav_tokenizer_model_path = os.path.join(MODEL_DIR, 'wavtokenizer_large_speech_320_24k.ckpt')

        # Initialize audio tokenizer
        audio_tokenizer = AudioTokenizer(
            hf_path,
            wav_tokenizer_model_path,
            wav_tokenizer_config_path
        )

        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            hf_path, 
             torch_dtype=torch.float32,  # Half precision
            low_cpu_mem_usage=True,  # Use low memory mode
            device_map='auto'  # Automatic device placement
        )
        
        # Optional: Enable model compilation for faster inference
        if torch.cuda.is_available():
            model = torch.compile(model)

        logger.info("Model and tokenizer initialized successfully!")
        return model, audio_tokenizer

    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        raise
    
def main():
    """
    Main application entry point with enhanced network configuration
    """
    # Get local IP
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    
    # Initialize model and tokenizer
    model, audio_tokenizer = initialize_model()
    
    # Create Flask app
    app = create_app(model, audio_tokenizer)
    
    # Run the application with more robust configuration
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=5000, 
        debug=True  # Enable debug for detailed error tracking
    )

if __name__ == '__main__':
    main()