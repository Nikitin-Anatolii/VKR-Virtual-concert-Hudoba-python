import librosa
import numpy as np
import json
from typing import Dict, List, Tuple
import os
import argparse
from pathlib import Path

# Audio analysis constants
WINDOW_SIZE = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100
BPM = 120  # Default BPM value

class AudioAnalyzer:
    def __init__(self, window_size: int = WINDOW_SIZE, hop_length: int = HOP_LENGTH):
        """
        Initialize the audio analyzer with analysis parameters.
        
        Args:
            window_size: Size of the FFT window
            hop_length: Number of samples between successive frames
        """
        self.window_size = window_size
        self.hop_length = hop_length

    def calculate_spectral_flux(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Calculate spectral flux for each second of audio.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Array of spectral flux values for each second
        """
        # Calculate spectrogram
        D = np.abs(librosa.stft(y, n_fft=self.window_size, hop_length=self.hop_length))
        
        # Calculate spectral flux
        flux = np.diff(D, axis=1)
        flux = np.sum(flux**2, axis=0)
        
        # Convert to per-second values
        seconds = int(len(y) / sr)
        flux_per_second = np.zeros(seconds)
        
        for i in range(seconds):
            start_frame = i * sr // self.hop_length
            end_frame = (i + 1) * sr // self.hop_length
            if end_frame > len(flux):
                end_frame = len(flux)
            if start_frame < len(flux):
                flux_per_second[i] = np.mean(flux[start_frame:end_frame])
        
        return flux_per_second

    def analyze_audio(self, file_path: str) -> Dict:
        """
        Analyze an audio file and return features with timestamps.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing timestamps and audio features
        """
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # Calculate duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate spectral flux
        spectral_flux = self.calculate_spectral_flux(y, sr)
        
        # Initialize arrays for per-second analysis
        seconds = np.arange(0, int(duration))
        
        # Calculate features for each second
        rms_values = []
        spectral_centroids = []
        
        for second in seconds:
            # Get the audio segment for this second
            start_sample = second * sr
            end_sample = (second + 1) * sr
            if end_sample > len(y):
                end_sample = len(y)
            
            segment = y[start_sample:end_sample]
            
            if len(segment) > 0:
                # Calculate RMS Energy
                rms = librosa.feature.rms(y=segment, frame_length=self.window_size, hop_length=self.hop_length)[0]
                rms_values.append(float(np.mean(rms)))
                
                # Calculate Spectral Centroid
                centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, 
                                                           n_fft=self.window_size, 
                                                           hop_length=self.hop_length)[0]
                spectral_centroids.append(float(np.mean(centroid)))
            else:
                rms_values.append(0.0)
                spectral_centroids.append(0.0)
        
        # Normalize all features to 0-1 range
        rms_normalized = librosa.util.normalize(np.array(rms_values))
        spectral_centroids_normalized = librosa.util.normalize(np.array(spectral_centroids))
        spectral_flux_normalized = librosa.util.normalize(spectral_flux)
        
        # Create result dictionary
        result = {
            "bpm": BPM,  # Using constant BPM
            "frames": []
        }
        
        # Add frame-by-frame analysis
        for i in range(len(seconds)):
            frame_data = {
                "spectral_flux": float(spectral_flux_normalized[i]),
                "energy": float(rms_normalized[i]),
                "spectral_centroid": float(spectral_centroids_normalized[i])
            }
            result["frames"].append(frame_data)
        
        return result

    def save_analysis(self, analysis: Dict, output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis: Analysis results dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze audio file and extract features')
    parser.add_argument('--input', '-i', type=str, help='Input audio file path')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine input file path
    if args.input:
        audio_file = args.input
    else:
        audio_file = os.path.join(current_dir, "input_audio.mp3")
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(current_dir, "audio_analysis.json")
    
    if os.path.exists(audio_file):
        print(f"Analyzing audio file: {audio_file}")
        # Analyze the audio
        analysis = analyzer.analyze_audio(audio_file)
        
        # Save the results
        analyzer.save_analysis(analysis, output_file)
        print(f"Analysis complete. Results saved to {output_file}")
        print(f"BPM: {BPM}")
    else:
        print(f"Audio file not found: {audio_file}")

if __name__ == "__main__":
    main() 