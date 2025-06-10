import librosa
import numpy as np
import os
from pathlib import Path
import random
import soundfile as sf
from scipy import signal
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TransitionDatasetGenerator:
    def __init__(self, songs_folder="training_songs", sample_rate=22050):
        self.songs_folder = Path(songs_folder)
        self.sample_rate = sample_rate
        self.songs = []
        self.dataset = []
        
    def load_all_songs(self):
        """Load all audio files from the songs folder"""
        print("Loading songs...")
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        
        for file_path in self.songs_folder.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                try:
                    # Fixed librosa.load call - removed extra arguments
                    audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
                    
                    # Only keep songs longer than 60 seconds
                    if len(audio) > 60 * self.sample_rate:
                        # Get tempo safely
                        try:
                            tempo = librosa.beat.tempo(audio)[0]
                        except:
                            tempo = 120.0  # Default tempo if detection fails
                        
                        self.songs.append({
                            'path': str(file_path),
                            'audio': audio,
                            'name': file_path.stem,
                            'tempo': tempo,
                            'key': self._estimate_key(audio)
                        })
                        print(f"✓ Loaded: {file_path.name} ({len(audio)/self.sample_rate:.1f}s, {tempo:.1f} BPM)")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.songs)} songs")
        return len(self.songs)
    
    def _estimate_key(self, audio):
        """Rough key estimation using chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            return np.argmax(np.mean(chroma, axis=1))
        except:
            return 0  # Default key if estimation fails
    
    def _apply_reverb(self, audio, room_size=0.3, damping=0.5):
        """Add reverb effect"""
        # Simple reverb using convolution with exponential decay
        reverb_length = int(0.3 * self.sample_rate)
        reverb_impulse = np.exp(-np.linspace(0, 5, reverb_length)) * np.random.randn(reverb_length) * 0.1
        reverbed = signal.convolve(audio, reverb_impulse, mode='same')
        return audio + reverbed * room_size
    
    def _apply_lowpass_sweep(self, audio, start_freq=8000, end_freq=200):
        """Apply a lowpass filter sweep"""
        sweep_length = len(audio)
        filtered_audio = np.zeros_like(audio)
        
        for i in range(0, sweep_length, 1024):
            # Interpolate cutoff frequency
            progress = i / sweep_length
            cutoff = start_freq * (1 - progress) + end_freq * progress
            
            # Apply butterworth filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))
            
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            chunk_end = min(i + 1024, sweep_length)
            filtered_audio[i:chunk_end] = signal.filtfilt(b, a, audio[i:chunk_end])
        
        return filtered_audio
    
    def _apply_highpass_sweep(self, audio, start_freq=50, end_freq=1000):
        """Apply a highpass filter sweep"""
        sweep_length = len(audio)
        filtered_audio = np.zeros_like(audio)
        
        for i in range(0, sweep_length, 1024):
            progress = i / sweep_length
            cutoff = start_freq * (1 - progress) + end_freq * progress
            
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))
            
            b, a = signal.butter(4, normalized_cutoff, btype='high')
            chunk_end = min(i + 1024, sweep_length)
            filtered_audio[i:chunk_end] = signal.filtfilt(b, a, audio[i:chunk_end])
        
        return filtered_audio
    
    def _create_echo_fade(self, audio, decay=0.3, delay_samples=None):
        """Create echo effect with fade"""
        if delay_samples is None:
            delay_samples = int(0.125 * self.sample_rate)  # 125ms delay
        
        echoed = np.zeros(len(audio) + delay_samples)
        echoed[:len(audio)] = audio
        
        # Add multiple echo taps
        for i, strength in enumerate([decay, decay*0.6, decay*0.3]):
            delay = delay_samples * (i + 1)
            if delay < len(echoed):
                echoed[delay:delay+len(audio)] += audio * strength
        
        return echoed[:len(audio)]
    
    def _create_stutter_effect(self, audio, stutter_length=0.125):
        """Create stutter/repeat effect"""
        stutter_samples = int(stutter_length * self.sample_rate)
        stuttered = []
        
        # Take the last bit and repeat it with decreasing volume
        stutter_chunk = audio[-stutter_samples:]
        for i in range(4):
            volume = 0.8 ** i
            stuttered.append(stutter_chunk * volume)
        
        return np.concatenate([audio[:-stutter_samples]] + stuttered)
    
    def _beat_match_audio(self, audio_a, audio_b, tolerance=5):
        """Simple beat matching - adjust tempo if they're close"""
        try:
            tempo_a = librosa.beat.tempo(audio_a)[0]
            tempo_b = librosa.beat.tempo(audio_b)[0]
            
            if abs(tempo_a - tempo_b) <= tolerance:
                # Stretch audio_b to match audio_a's tempo
                stretch_factor = tempo_a / tempo_b
                audio_b_matched = librosa.effects.time_stretch(audio_b, rate=stretch_factor)
                return audio_b_matched
        except:
            # If beat matching fails, return original
            pass
        
        return audio_b
    
    def create_transition(self, song_a, song_b, style='random'):
        """Create a single transition between two songs"""
        # Extract segments
        segment_length = 15 * self.sample_rate  # 15 seconds
        ending_a = song_a['audio'][-segment_length:]
        beginning_b = song_b['audio'][:segment_length]
        
        # Beat match if tempos are close
        beginning_b = self._beat_match_audio(ending_a, beginning_b)
        beginning_b = beginning_b[:segment_length]  # Trim back to original length
        
        if style == 'random':
            style = random.choice(['crossfade', 'reverb_out', 'filter_sweep', 'echo_fade', 
                                 'highpass_build', 'stutter_out', 'dramatic_cut'])
        
        fade_duration = random.uniform(3, 8)  # 3-8 second transitions
        fade_samples = int(fade_duration * self.sample_rate)
        
        if style == 'crossfade':
            # Simple linear crossfade
            overlap = min(fade_samples, len(ending_a), len(beginning_b))
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            
            transition = np.zeros(len(ending_a) + len(beginning_b) - overlap)
            transition[:len(ending_a)] = ending_a
            transition[-len(beginning_b):] += beginning_b
            transition[len(ending_a)-overlap:len(ending_a)] *= fade_out
            transition[len(ending_a)-overlap:len(ending_a)] += beginning_b[:overlap] * fade_in
            
        elif style == 'reverb_out':
            # Add reverb to ending, then crossfade
            reverbed_ending = self._apply_reverb(ending_a, room_size=0.6)
            overlap = fade_samples
            transition = np.zeros(len(reverbed_ending) + len(beginning_b) - overlap)
            transition[:len(reverbed_ending)] = reverbed_ending
            transition[-len(beginning_b):] += beginning_b
            
            # Crossfade the overlap
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            transition[len(reverbed_ending)-overlap:len(reverbed_ending)] *= fade_out
            transition[len(reverbed_ending)-overlap:len(reverbed_ending)] += beginning_b[:overlap] * fade_in
            
        elif style == 'filter_sweep':
            # Apply lowpass sweep to ending
            filtered_ending = self._apply_lowpass_sweep(ending_a)
            overlap = fade_samples
            transition = np.zeros(len(filtered_ending) + len(beginning_b) - overlap)
            transition[:len(filtered_ending)] = filtered_ending
            transition[-len(beginning_b):] += beginning_b
            
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            transition[len(filtered_ending)-overlap:len(filtered_ending)] *= fade_out
            transition[len(filtered_ending)-overlap:len(filtered_ending)] += beginning_b[:overlap] * fade_in
            
        elif style == 'echo_fade':
            # Echo effect on ending
            echoed_ending = self._create_echo_fade(ending_a, decay=0.4)
            overlap = fade_samples
            transition = np.zeros(len(echoed_ending) + len(beginning_b) - overlap)
            transition[:len(echoed_ending)] = echoed_ending
            transition[-len(beginning_b):] += beginning_b
            
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            transition[len(echoed_ending)-overlap:len(echoed_ending)] *= fade_out
            transition[len(echoed_ending)-overlap:len(echoed_ending)] += beginning_b[:overlap] * fade_in
            
        elif style == 'highpass_build':
            # Highpass sweep on beginning
            filtered_beginning = self._apply_highpass_sweep(beginning_b, start_freq=50, end_freq=20)
            overlap = fade_samples
            transition = np.zeros(len(ending_a) + len(filtered_beginning) - overlap)
            transition[:len(ending_a)] = ending_a
            transition[-len(filtered_beginning):] += filtered_beginning
            
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            transition[len(ending_a)-overlap:len(ending_a)] *= fade_out
            transition[len(ending_a)-overlap:len(ending_a)] += filtered_beginning[:overlap] * fade_in
            
        elif style == 'stutter_out':
            # Stutter effect on ending
            stuttered_ending = self._create_stutter_effect(ending_a)
            overlap = fade_samples
            transition = np.zeros(len(stuttered_ending) + len(beginning_b) - overlap)
            transition[:len(stuttered_ending)] = stuttered_ending
            transition[-len(beginning_b):] += beginning_b
            
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            transition[len(stuttered_ending)-overlap:len(stuttered_ending)] *= fade_out
            transition[len(stuttered_ending)-overlap:len(stuttered_ending)] += beginning_b[:overlap] * fade_in
            
        elif style == 'dramatic_cut':
            # Sharp cut with brief silence
            silence_duration = random.uniform(0.1, 0.5)  # 100-500ms silence
            silence_samples = int(silence_duration * self.sample_rate)
            
            # Fade out ending quickly
            quick_fade = int(0.1 * self.sample_rate)
            ending_a[-quick_fade:] *= np.linspace(1, 0, quick_fade)
            
            # Fade in beginning quickly  
            beginning_b[:quick_fade] *= np.linspace(0, 1, quick_fade)
            
            transition = np.concatenate([
                ending_a,
                np.zeros(silence_samples),
                beginning_b
            ])
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(transition))
        if max_val > 0:
            transition = transition / max_val * 0.9
        
        return transition, style
    
    def generate_dataset(self, num_samples=1000, save_audio=False):
        """Generate the complete dataset"""
        if len(self.songs) < 2:
            raise ValueError("Need at least 2 songs to create transitions!")
        
        print(f"Generating {num_samples} transition examples...")
        
        # Create output directories
        if save_audio:
            os.makedirs("dataset/audio", exist_ok=True)
        os.makedirs("dataset", exist_ok=True)
        
        dataset = []
        style_counts = {}
        
        for i in tqdm(range(num_samples)):
            try:
                # Randomly select two different songs
                song_a, song_b = random.sample(self.songs, 2)
                
                # Create transition
                transition_audio, style = self.create_transition(song_a, song_b)
                
                # Track style distribution
                style_counts[style] = style_counts.get(style, 0) + 1
                
                # Create data entry
                data_entry = {
                    'id': i,
                    'song_a': song_a['name'],
                    'song_b': song_b['name'],
                    'song_a_tempo': float(song_a['tempo']),
                    'song_b_tempo': float(song_b['tempo']),
                    'song_a_key': int(song_a['key']),
                    'song_b_key': int(song_b['key']),
                    'transition_style': style,
                    'transition_duration': len(transition_audio) / self.sample_rate,
                    'audio_file': f"transition_{i:06d}.wav" if save_audio else None
                }
                
                dataset.append(data_entry)
                
                # Save audio file if requested
                if save_audio:
                    sf.write(f"dataset/audio/transition_{i:06d}.wav", 
                            transition_audio, self.sample_rate)
                    
            except Exception as e:
                print(f"Error creating transition {i}: {e}")
                continue
        
        # Save dataset metadata
        with open("dataset/dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save style distribution
        print(f"\nStyle distribution:")
        for style, count in style_counts.items():
            print(f"  {style}: {count} ({count/len(dataset)*100:.1f}%)")
        
        print(f"\nDataset saved to dataset/")
        print(f"Total samples: {len(dataset)}")
        
        return dataset

# Usage
if __name__ == "__main__":
    # Initialize generator
    generator = TransitionDatasetGenerator()
    
    # Load songs
    num_songs = generator.load_all_songs()
    
    if num_songs == 0:
        print("ERROR: No songs found in 'transition_songs' folder!")
        print("Make sure you have audio files (.mp3, .wav, .flac, .m4a, .ogg) in the folder.")
        exit(1)
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_samples=500,   # Start with fewer samples to test
        save_audio=True    # Set to False if you don't want to save audio files
    )
    
    print("Dataset generation complete!")
    print("Next steps:")
    print("1. Check dataset/dataset.json for metadata")
    print("2. Listen to some examples in dataset/audio/")
    print("3. Use this data to train your transition generation model")