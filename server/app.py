from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import numpy as np
import utils
import logging
import torchaudio
import librosa

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Set a custom temporary directory
TEMP_FOLDER = 'temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

tempfile.tempdir = TEMP_FOLDER

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and feature extractor
MODEL_PATH = './models/CW_ham'
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model.eval()

def extract_audio(file_path):
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_path = os.path.splitext(file_path)[0] + '.wav'
        audio.write_audiofile(audio_path)
        video.close()
        return audio_path
    return file_path

def preprocess_audio(file_path, segment_length=16000, hop_length=8000):
    """
    Preprocess audio using the same techniques as in fine-tuning.
    segment_length=16000 represents 1 second at 16kHz
    hop_length=8000 represents 0.5 second overlap
    """
    # Get the sample rate
    metadata = torchaudio.info(file_path)
    sr = metadata.sample_rate
    
    # Load the audio
    audio, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Convert to numpy array for processing
    audio_np = audio.squeeze().numpy()
    
    # Apply pre-emphasis filter to reduce noise
    audio_np = librosa.effects.preemphasis(audio_np)
    
    # Simple noise reduction by removing low amplitude noise
    noise_threshold = 0.005
    audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
    
    # Convert back to torch tensor
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)
    
    audio = audio.squeeze().numpy()
    
    # Split audio into overlapping segments
    segments = []
    for start in range(0, len(audio), hop_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    
    return segments, len(audio) / 16000

@app.route('/api/censor', methods=['POST'])
def censor_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # Extract audio if it's a video file
        audio_path = extract_audio(temp_path)

        # Process the audio
        segments, duration = preprocess_audio(audio_path)
        results = []
        
        logging.debug(f"Processing {len(segments)} segments")
        
        # Labels matching the fine-tuned model
        labels = ["none", "เย็ด", "กู", "มึง", "เหี้ย", "ควย", "สวะ", "หี", "แตด"]
        
        for i, segment in enumerate(segments):
            prediction, probabilities = utils.predict(model, feature_extractor, segment)
            logging.debug(f"Segment {i}: prediction={prediction} ({labels[prediction]}), probabilities={probabilities}")
            
            # Check all profanity classes
            for class_idx, prob in enumerate(probabilities):
                if class_idx != 0 and prob > 0.4:  # Skip 'none' class and use threshold
                    start_time = i * 0.5  # 0.5 seconds hop length
                    end_time = min(start_time + 1, duration)
                    detected_word = labels[class_idx]
                    results.append((start_time, end_time, prob, detected_word))
                    logging.debug(f"Profanity '{detected_word}' detected at segment {i}: {start_time}s to {end_time}s (prob: {prob})")

        # Merge and censor detections
        merged_results = utils.merge_detections(results)
        logging.debug(f"Merged results: {merged_results}")
        
        if merged_results:
            # Convert merged results to format expected by censor_audio (only need start, end, prob)
            censor_results = [(start, end, prob) for start, end, prob, _ in merged_results]
            
            # Reference your censor_audio function
            censored_path = utils.censor_audio(audio_path, censor_results)
            
            # Log the path and check if the file exists
            logging.debug(f"Censored audio path: {censored_path}")
            if not os.path.exists(censored_path):
                logging.error(f"Censored audio file not found: {censored_path}")
                raise FileNotFoundError(f"Censored audio file not found: {censored_path}")

            # Move the censored file to the processed folder
            processed_filename = f'censored_{filename}'
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            
            # Remove existing file if it exists
            if os.path.exists(processed_path):
                os.remove(processed_path)
            
            # If it's an audio file, move it to processed folder
            if temp_path == audio_path:
                os.rename(censored_path, processed_path)
                return_path = processed_path
            else:
                # If original was video, merge censored audio back
                video = VideoFileClip(temp_path)
                censored_audio = AudioFileClip(censored_path)
                final_video = video.set_audio(censored_audio)
                output_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                final_video.write_videofile(output_path)
                video.close()
                censored_audio.close()
                return_path = output_path

            # Convert numpy float32 to regular Python float for JSON serialization
            json_merged_results = [
                (float(start), float(end), float(prob), word)
                for start, end, prob, word in merged_results
            ]

            # Return the URL to download the censored file
            return jsonify({
                'censoredUrl': f'/download/{os.path.basename(return_path)}',
                'censoredSegments': json_merged_results  # Now contains regular Python floats
            })
        else:
            # If no profanity detected, just copy the original file
            processed_filename = f'censored_{filename}'
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            if temp_path == audio_path:
                import shutil
                shutil.copy2(audio_path, processed_path)
                return_path = processed_path
            else:
                video = VideoFileClip(temp_path)
                output_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                video.write_videofile(output_path)
                video.close()
                return_path = output_path

            return jsonify({
                'censoredUrl': f'/download/{os.path.basename(return_path)}'
            })

    except Exception as e:
        logging.error(f"Error in /api/censor: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if audio_path != temp_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(PROCESSED_FOLDER, filename),
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(port=8080, debug=True) 