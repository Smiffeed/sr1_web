import torch
import torchaudio
import numpy as np
import tempfile
import os
import librosa
import soundfile as sf
from typing import Tuple, List
from pydub import AudioSegment

def preprocess_audio(file_path: str, segment_duration: float = 0.5) -> Tuple[List[np.ndarray], float]:
    """
    Preprocess audio file into segments for profanity detection using the same
    preprocessing steps as in model fine-tuning.
    
    Args:
        file_path: Path to the audio file
        segment_duration: Duration of each segment in seconds
    
    Returns:
        Tuple containing:
        - List of preprocessed audio segments
        - Total duration of the audio in seconds
    """
    try:
        # Load the audio file and get metadata
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        duration = metadata.num_frames / sr
        
        # Load the full audio
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Convert to numpy array for processing
        audio_np = audio.squeeze().numpy()
        
        # Apply Hamming window
        window_length = len(audio_np)
        hamming_window = np.hamming(window_length)
        audio_np = audio_np * hamming_window
        
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
            sr = 16000
        
        # Normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        
        # Calculate segment size in samples
        segment_size = int(segment_duration * sr)
        
        # Split audio into segments with overlap
        hop_length = segment_size // 2  # 50% overlap
        segments = []
        for start in range(0, audio.shape[1], hop_length):
            end = start + segment_size
            segment = audio[:, start:end]
            
            # Pad if segment is shorter than segment_size
            if segment.shape[1] < segment_size:
                segment = torch.nn.functional.pad(segment, (0, segment_size - segment.shape[1]))
            
            # Convert to numpy and append
            segments.append(segment.squeeze().numpy())
        
        return segments, duration
        
    except Exception as e:
        print(f"Error in preprocess_audio for {file_path}: {str(e)}")
        raise

def remove_background_noise(input_path: str, output_path: str) -> None:
    """
    Remove background noise from audio file.
    You can implement this based on your existing background noise removal code.
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None)
        
        # Apply noise reduction
        # This is a simple example - you might want to use your existing noise reduction code
        D = librosa.stft(y)
        D_denoised = librosa.decompose.nn_filter(D,
                                            aggregate=np.median,
                                            metric='cosine',
                                            width=3)
        y_denoised = librosa.istft(D_denoised)
        
        # Save the denoised audio
        sf.write(output_path, y_denoised, sr)
        
    except Exception as e:
        print(f"Error in remove_background_noise: {str(e)}")
        raise

def predict(model, feature_extractor, audio):
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0].item(), probabilities[0].numpy()

def merge_detections(detections, threshold=0.4, min_gap=0.5):
    """
    Merge detections with improved handling of different profanity words
    threshold: minimum probability to keep the detection
    min_gap: minimum gap in seconds between detections of the same word
    """
    if not detections:
        return []
        
    # Sort by start time
    detections = sorted(detections, key=lambda x: x[0])
    
    merged = []
    current_group = list(detections[0])
    
    for detection in detections[1:]:
        start, end, prob, word = detection
        # If this detection starts after the current group ends (plus min_gap)
        # or if it's a different word
        if start - current_group[1] >= min_gap or word != current_group[3]:
            if current_group[2] >= threshold:
                merged.append(tuple(current_group))
            current_group = [start, end, prob, word]
        else:
            # Merge only if it's the same word and has higher probability
            if prob > current_group[2]:
                current_group[1] = end
                current_group[2] = prob
            else:
                current_group[1] = max(current_group[1], end)
    
    # Add the last group if it meets the threshold
    if current_group[2] >= threshold:
        merged.append(tuple(current_group))
    
    return merged

def censor_audio(file_path, detections):
    try:
        audio = AudioSegment.from_wav(file_path)
        
        # Check if beep.wav exists
        beep_path = "beep.wav"
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep sound file not found at {beep_path}")
           
        beep = AudioSegment.from_wav(beep_path)

        # Sort detections by start time
        detections = sorted(detections, key=lambda x: x[0])
        
        # Create a new audio segment
        censored_audio = audio[:int(detections[0][0] * 1000)] if detections else audio

        for i, (start, end, _) in enumerate(detections):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            segment_duration = end_ms - start_ms
            
            # Adjust beep duration to match the censored segment
            adjusted_beep = beep[:segment_duration] if len(beep) > segment_duration else beep + AudioSegment.silent(duration=segment_duration - len(beep))
            
            # Add beep
            censored_audio += adjusted_beep
            
            # Add clean audio until next detection or end
            next_start = int(detections[i+1][0] * 1000) if i < len(detections)-1 else len(audio)
            censored_audio += audio[end_ms:next_start]

        censored_file_path = file_path.rsplit('.', 1)[0] + '_censored.wav'
        censored_audio.export(censored_file_path, format="wav")
        return censored_file_path
        
    except Exception as e:
        print(f"Error in censor_audio: {str(e)}")
        raise