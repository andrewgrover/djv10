import pygame
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import threading
import time
from train_model import TransitionGenerator

class AIRealTimeDJ:
    def __init__(self, music_folder="dj_music", model_path="transition_model_final.pth"):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        self.music_folder = Path(music_folder)
        self.current_song = None
        self.next_song = None
        self.is_playing = False
        self.transition_active = False
        
        # Load AI transition generator
        self.transition_generator = TransitionGenerator(model_path)
        
        # Load music library
        self.music_library = self.load_music_library()
        self.current_index = 0
        
        print(f"Loaded {len(self.music_library)} songs")
        
    def load_music_library(self):
        """Load all music files from the music folder"""
        library = []
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        
        for file_path in self.music_folder.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                try:
                    # Get basic info about the track
                    audio, sr = librosa.load(str(file_path), sr=22050, duration=30)  # Load first 30s for analysis
                    tempo = librosa.beat.tempo(audio)[0]
                    
                    library.append({
                        'path': str(file_path),
                        'name': file_path.stem,
                        'tempo': tempo,
                        'duration': librosa.get_duration(filename=str(file_path))
                    })
                    print(f"Added: {file_path.name} - {tempo:.1f} BPM")
                except Exception as e:
                    print(f"Failed to load {file_path.name}: {e}")
        
        return library
    
    def play_song(self, song_index):
        """Play a song by index"""
        if 0 <= song_index < len(self.music_library):
            song = self.music_library[song_index]
            pygame.mixer.music.load(song['path'])
            pygame.mixer.music.play()
            self.current_song = song
            self.current_index = song_index
            self.is_playing = True
            print(f"Now playing: {song['name']}")
    
    def stop_music(self):
        """Stop current music"""
        pygame.mixer.music.stop()
        self.is_playing = False
    
    def pause_music(self):
        """Pause/unpause music"""
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
        else:
            pygame.mixer.music.unpause()
            self.is_playing = True
    
    def next_song(self):
        """Play next song"""
        next_index = (self.current_index + 1) % len(self.music_library)
        self.play_song(next_index)
    
    def previous_song(self):
        """Play previous song"""
        prev_index = (self.current_index - 1) % len(self.music_library)
        self.play_song(prev_index)
    
    def smart_transition_to_next(self, style='random'):
        """Create AI transition to next song"""
        if not self.current_song or self.transition_active:
            return
        
        self.transition_active = True
        
        try:
            # Get current and next song
            current_song = self.music_library[self.current_index]
            next_index = (self.current_index + 1) % len(self.music_library)
            next_song = self.music_library[next_index]
            
            print(f"Creating AI transition: {current_song['name']} -> {next_song['name']}")
            
            # Calculate musical differences
            tempo_diff = next_song['tempo'] - current_song['tempo']
            key_diff = 0  # We'd need to calculate this properly
            
            # Generate transition
            if style == 'random':
                style = np.random.choice(['crossfade', 'reverb_out', 'filter_sweep', 'echo_fade'])
            
            transition_audio = self.transition_generator.generate_transition(
                style=style, 
                tempo_diff=tempo_diff, 
                key_diff=key_diff
            )
            
            # Save transition temporarily
            transition_path = "temp_transition.wav"
            sf.write(transition_path, transition_audio, 22050)
            
            # Stop current song and play transition
            pygame.mixer.music.stop()
            pygame.mixer.music.load(transition_path)
            pygame.mixer.music.play()
            
            # Wait for transition to finish, then play next song
            transition_duration = len(transition_audio) / 22050
            
            def play_next_after_transition():
                time.sleep(transition_duration)
                self.play_song(next_index)
                self.transition_active = False
            
            threading.Thread(target=play_next_after_transition, daemon=True).start()
            
            print(f"Applied {style} transition ({transition_duration:.1f}s)")
            
        except Exception as e:
            print(f"Transition failed: {e}")
            self.next_song()  # Fallback to regular next
            self.transition_active = False
    
    def get_music_position(self):
        """Get current playback position"""
        return pygame.mixer.music.get_pos() / 1000.0  # Convert to seconds
    
    def run_console_interface(self):
        """Run simple console-based DJ interface"""
        print("\n=== AI DJ CONSOLE ===")
        print("Commands:")
        print("  play <number> - Play song by number")
        print("  next - Next song")
        print("  prev - Previous song")
        print("  pause - Pause/unpause")
        print("  stop - Stop music")
        print("  transition <style> - AI transition to next song")
        print("    Styles: crossfade, reverb_out, filter_sweep, echo_fade, random")
        print("  list - List all songs")
        print("  status - Show current status")
        print("  quit - Exit")
        print()
        
        # List available songs
        print("Available songs:")
        for i, song in enumerate(self.music_library):
            print(f"  {i}: {song['name']} ({song['tempo']:.1f} BPM)")
        print()
        
        while True:
            try:
                command = input("DJ> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'list':
                    for i, song in enumerate(self.music_library):
                        marker = "♪" if i == self.current_index and self.is_playing else " "
                        print(f"  {marker} {i}: {song['name']} ({song['tempo']:.1f} BPM)")
                
                elif command == 'next':
                    self.next_song()
                
                elif command == 'prev':
                    self.previous_song()
                
                elif command == 'pause':
                    self.pause_music()
                    print("Paused" if not self.is_playing else "Resumed")
                
                elif command == 'stop':
                    self.stop_music()
                    print("Stopped")
                
                elif command == 'status':
                    if self.current_song:
                        status = "Playing" if self.is_playing else "Paused"
                        print(f"Status: {status}")
                        print(f"Current: {self.current_song['name']}")
                        print(f"Tempo: {self.current_song['tempo']:.1f} BPM")
                    else:
                        print("Status: No song loaded")
                
                elif command.startswith('play '):
                    try:
                        song_num = int(command.split()[1])
                        self.play_song(song_num)
                    except (ValueError, IndexError):
                        print("Usage: play <song_number>")
                
                elif command.startswith('transition'):
                    parts = command.split()
                    style = parts[1] if len(parts) > 1 else 'random'
                    if style not in ['crossfade', 'reverb_out', 'filter_sweep', 'echo_fade', 'random']:
                        print("Invalid style. Use: crossfade, reverb_out, filter_sweep, echo_fade, random")
                    else:
                        self.smart_transition_to_next(style)
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.stop_music()
        print("DJ session ended!")

# Auto DJ Mode
class AutoDJ(AIRealTimeDJ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_mode = False
        self.transition_styles = ['crossfade', 'reverb_out', 'filter_sweep', 'echo_fade']
    
    def start_auto_mode(self, transition_interval=30):
        """Start automatic DJ mode with AI transitions"""
        self.auto_mode = True
        
        def auto_dj_loop():
            while self.auto_mode:
                if self.is_playing and not self.transition_active:
                    # Check if we should transition (based on song progress)
                    # For now, just transition every X seconds
                    time.sleep(transition_interval)
                    if self.auto_mode:
                        style = np.random.choice(self.transition_styles)
                        print(f"[AUTO DJ] Transitioning with {style}")
                        self.smart_transition_to_next(style)
                else:
                    time.sleep(1)
        
        threading.Thread(target=auto_dj_loop, daemon=True).start()
        print(f"Auto DJ mode started (transitions every {transition_interval}s)")
    
    def stop_auto_mode(self):
        """Stop automatic DJ mode"""
        self.auto_mode = False
        print("Auto DJ mode stopped")

if __name__ == "__main__":
    # Make sure you have a 'dj_music' folder with music files
    # and a trained model 'transition_model_final.pth'
    
    print("Starting AI DJ...")
    
    try:
        dj = AIRealTimeDJ()
        
        if len(dj.music_library) == 0:
            print("No music found! Add music files to 'dj_music' folder")
        else:
            # Start with first song
            dj.play_song(0)
            
            # Run the interface
            dj.run_console_interface()
    
    except FileNotFoundError:
        print("Model file 'transition_model_final.pth' not found!")
        print("Train your model first using the training script.")
    except Exception as e:
        print(f"Error starting DJ: {e}")
        print("Make sure you have:")
        print("1. A 'dj_music' folder with music files")
        print("2. A trained model 'transition_model_final.pth'")
        print("3. All required packages installed")