import librosa
import numpy as np
import json
from typing import Dict, List, Tuple
import os

class AudioAnalyzer:
    def __init__(self, window_size: int = 2048, hop_length: int = 512):
        """
        Initialize the audio analyzer with analysis parameters.
        
        Args:
            window_size: Size of the FFT window
            hop_length: Number of samples between successive frames
        """
        self.window_size = window_size
        self.hop_length = hop_length

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
        
        # Calculate time points for each frame
        times = librosa.times_like(y, sr=sr, hop_length=self.hop_length)
        
        # Calculate RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.window_size, hop_length=self.hop_length)[0]
        
        # Calculate Tempo/BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Calculate Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                             n_fft=self.window_size, 
                                                             hop_length=self.hop_length)[0]
        
        # Normalize features
        rms_normalized = librosa.util.normalize(rms)
        spectral_centroids_normalized = librosa.util.normalize(spectral_centroids)
        
        # Ensure all arrays have the same length
        min_length = min(len(times), len(rms_normalized), len(spectral_centroids_normalized))
        times = times[:min_length]
        rms_normalized = rms_normalized[:min_length]
        spectral_centroids_normalized = spectral_centroids_normalized[:min_length]
        
        # Create result dictionary
        result = {
            "tempo_bpm": float(tempo[0] if isinstance(tempo, np.ndarray) else tempo),
            "frames": []
        }
        
        # Add frame-by-frame analysis
        for i in range(min_length):
            frame_data = {
                "timestamp": float(times[i]),
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
    # Example usage
    analyzer = AudioAnalyzer()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Example audio file path (you'll need to provide your own audio file)
    audio_file = os.path.join(current_dir, "input_audio.mp3")
    
    if os.path.exists(audio_file):
        # Analyze the audio
        analysis = analyzer.analyze_audio(audio_file)
        
        # Save the results
        output_file = os.path.join(current_dir, "audio_analysis.json")
        analyzer.save_analysis(analysis, output_file)
        print(f"Analysis complete. Results saved to {output_file}")
    else:
        print(f"Audio file not found: {audio_file}")

if __name__ == "__main__":
    main() 