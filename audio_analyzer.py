import librosa
import numpy as np
import json
from typing import Dict, List, Tuple
import os
import argparse
from pathlib import Path

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

    def analyze_bpm(self, y: np.ndarray, sr: int) -> float:
        """
        Analyze BPM using multiple methods for better accuracy.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Estimated BPM value
        """
        # Method 1: Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm1 = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Method 2: Dynamic programming beat tracker
        tempo2, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Method 3: Autocorrelation
        y_harmonic, _ = librosa.effects.hpss(y)
        tempo3, _ = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        
        # Combine results (weighted average)
        bpm = (bpm1 + tempo2 + tempo3) / 3
        return bpm

    def check_audio_quality(self, y: np.ndarray, sr: int) -> Dict:
        """
        Check audio quality and return metrics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with quality metrics
        """
        # Calculate signal-to-noise ratio
        noise_floor = np.mean(np.abs(y[y < np.percentile(y, 10)]))
        signal_level = np.mean(np.abs(y[y > np.percentile(y, 90)]))
        snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Calculate dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(y)) / (np.min(np.abs(y[y > 0])) + 1e-10))
        
        return {
            "signal_to_noise_ratio": float(snr),
            "dynamic_range": float(dynamic_range),
            "sample_rate": sr,
            "duration": float(librosa.get_duration(y=y, sr=sr))
        }

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
        
        # Check audio quality
        quality_metrics = self.check_audio_quality(y, sr)
        
        # Calculate duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Initialize arrays for per-second analysis
        seconds = np.arange(0, int(duration))
        
        # Calculate features for each second
        rms_values = []
        spectral_centroids = []
        bpm_values = []
        
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
                
                # Calculate BPM for this second
                bpm = self.analyze_bpm(segment, sr)
                bpm_values.append(float(bpm))
            else:
                rms_values.append(0.0)
                spectral_centroids.append(0.0)
                bpm_values.append(0.0)
        
        # Normalize all features to 0-1 range
        rms_normalized = librosa.util.normalize(np.array(rms_values))
        spectral_centroids_normalized = librosa.util.normalize(np.array(spectral_centroids))
        bpm_normalized = librosa.util.normalize(np.array(bpm_values))
        
        # Create result dictionary
        result = {
            "audio_quality": quality_metrics,
            "frames": []
        }
        
        # Add frame-by-frame analysis
        for i in range(len(seconds)):
            frame_data = {
                "bpm": float(bpm_normalized[i]),
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
        
        # Print quality metrics
        print("\nAudio Quality Metrics:")
        print(f"Signal-to-Noise Ratio: {analysis['audio_quality']['signal_to_noise_ratio']:.2f} dB")
        print(f"Dynamic Range: {analysis['audio_quality']['dynamic_range']:.2f} dB")
        print(f"Sample Rate: {analysis['audio_quality']['sample_rate']} Hz")
        print(f"Duration: {analysis['audio_quality']['duration']:.2f} seconds")
    else:
        print(f"Audio file not found: {audio_file}")

if __name__ == "__main__":
    main() 