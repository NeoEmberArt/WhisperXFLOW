#!/usr/bin/env python3
"""
WhisperX Command Line Interface
A script to manage WhisperX models and transcribe audio files.
"""

import os
import sys
import subprocess
import json
import venv
from pathlib import Path
import warnings
import time
import gc
import io

# Fix for Windows console encoding issues
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

class WhisperXCLI:
    def __init__(self):
        self.venv_path = Path("whisperx_env")
        self.python_path = self.venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"
        self.pip_path = self.venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
        self.whisperx = None
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.current_model_name = None
        self.device = "cpu"
        self.compute_type = "int8"
        
        # Use ASCII symbols instead of emojis for Windows compatibility
        self.use_ascii = os.name == "nt"
        
        # Available models with approximate sizes
        self.available_models = {
            "tiny": "~39 MB",
            "tiny.en": "~39 MB", 
            "base": "~74 MB",
            "base.en": "~74 MB",
            "small": "~244 MB",
            "small.en": "~244 MB",
            "medium": "~769 MB",
            "medium.en": "~769 MB",
            "large": "~1550 MB",
            "large-v1": "~1550 MB",
            "large-v2": "~1550 MB",
            "large-v3": "~1550 MB"
        }
    
    # Helper methods to handle emoji vs ASCII
    def emoji(self, emoji_char, ascii_alt):
        return ascii_alt if self.use_ascii else emoji_char
        
    def setup_environment(self):
        """Create virtual environment if it doesn't exist and install WhisperX"""
        try:
            print("Setting up environment...")
            print("progress=5")
            
            if not self.venv_path.exists():
                print("Creating virtual environment...")
                venv.create(self.venv_path, with_pip=True)
                print("progress=20")
            else:
                print("Virtual environment exists")
                print("progress=20")
            
            # Check if WhisperX is installed
            print("Checking WhisperX installation...")
            result = subprocess.run([str(self.python_path), "-c", "import whisperx"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("Installing WhisperX... (this may take several minutes)")
                print("progress=30")
                process = subprocess.run([str(self.pip_path), "install", "whisperx"], 
                                       capture_output=True, text=True)
                if process.returncode != 0:
                    print(f"Error installing WhisperX: {process.stderr}")
                    return False
                print("progress=60")
            else:
                print("WhisperX already installed")
                print("progress=60")
            
            # Import WhisperX
            print("Importing WhisperX...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import whisperx
            self.whisperx = whisperx
            print("Environment setup complete!")
            print("progress=100")
            return True
            
        except Exception as e:
            print(f"Error setting up environment: {e}")
            return False
    
    def list_models(self):
        """List available models and their sizes"""
        print("\nAvailable WhisperX models:")
        print("-" * 50)
        for model, size in self.available_models.items():
            status = " (loaded & ready)" if model == self.current_model_name else ""
            print(f"{model:<15} {size:<10} {status}")
        print("-" * 50)
        if self.current_model_name:
            print(f"Currently loaded: {self.current_model_name} - Ready for fast transcription!")
        else:
            print("No model loaded. Use load-model(model_name) to prepare for transcription.")
    
    def load_model(self, model_name):
        """Load a WhisperX model and preload all components for fastest transcription"""
        # Remove quotes if present
        model_name = model_name.strip('"\'')
        
        if model_name not in self.available_models:
            print(f"{self.emoji('❌', 'X')} Error: Model '{model_name}' not found.")
            print("Available models:", ", ".join(self.available_models.keys()))
            return False
        
        if self.current_model_name == model_name and self.model is not None:
            print(f"{self.emoji('✅', '√')} Model '{model_name}' is already loaded and ready!")
            return True
        
        try:
            print(f"{self.emoji('🔄', '*')} Loading model '{model_name}' ({self.available_models[model_name]})...")
            print("This will preload everything for fastest transcription speed.")
            print("progress=5")
            
            # Clear previous models to free memory
            if self.model is not None:
                print("Clearing previous model from memory...")
                del self.model
                del self.align_model
                del self.align_metadata
                gc.collect()
                print("progress=10")
            
            print(f"{self.emoji('📥', '>')} Downloading/loading main transcription model...")
            print("(First-time downloads may take several minutes)")
            print("progress=15")
            
            # Load main transcription model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = self.whisperx.load_model(
                    model_name, 
                    self.device, 
                    compute_type=self.compute_type
                )
            print("progress=50")
            print(f"{self.emoji('✅', '√')} Main model loaded!")
            
            # Preload alignment model for common languages
            print(f"{self.emoji('📥', '>')} Preloading alignment model for word-level timestamps...")
            try:
                # Default to English alignment model for .en models, auto-detect for others
                default_lang = "en" if model_name.endswith(".en") else "en"
                print(f"Loading alignment model for language: {default_lang}")
                print("progress=70")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.align_model, self.align_metadata = self.whisperx.load_align_model(
                        language_code=default_lang, 
                        device=self.device
                    )
                print("progress=90")
                print(f"{self.emoji('✅', '√')} Alignment model preloaded!")
                
            except Exception as align_error:
                print(f"{self.emoji('⚠️', '!')} Warning: Could not preload alignment model: {align_error}")
                print("Word-level timestamps will be loaded during transcription if needed.")
                self.align_model = None
                self.align_metadata = None
            
            self.current_model_name = model_name
            print("progress=100")
            print(f"{self.emoji('🎉', '!')} Model '{model_name}' fully loaded and optimized for fast transcription!")
            print(f"{self.emoji('💡', 'i')} You can now use transcribe-audio() for maximum speed.")
            return True
            
        except Exception as e:
            print(f"{self.emoji('❌', 'X')} Error loading model '{model_name}': {e}")
            print("\nTroubleshooting:")
            print("- Check your internet connection for downloads")
            print("- Ensure sufficient disk space")
            print("- Try a smaller model first (e.g., tiny.en)")
            print("- Restart the script if memory issues occur")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file using the preloaded model for maximum speed"""
        # Remove quotes if present
        audio_path = audio_path.strip('"\'')
        
        if self.model is None:
            print(f"{self.emoji('❌', 'X')} Error: No model loaded!")
            print(f"{self.emoji('💡', 'i')} Use load-model(model_name) first to prepare for transcription.")
            print("Example: load-model(tiny.en)")
            return
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"{self.emoji('❌', 'X')} Error: Audio file not found: {audio_path}")
            print(f"{self.emoji('💡', 'i')} Make sure the file path is correct and the file exists.")
            print("Supported formats: .wav, .mp3, .m4a, .flac, .ogg, .wma")
            return
        
        start_time = time.time()
        
        try:
            print(f"{self.emoji('🎵', '#')} Transcribing: {audio_file.name}")
            print(f"{self.emoji('📊', '>')} Using model: {self.current_model_name}")
            print("progress=5")
            
            # Load audio (optimized)
            print(f"{self.emoji('📂', '>')} Loading audio file...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = self.whisperx.load_audio(str(audio_file))
            
            audio_duration = len(audio) / 16000  # WhisperX uses 16kHz
            print(f"{self.emoji('⏱️', '>')} Audio duration: {audio_duration:.1f} seconds")
            print("progress=15")
            
            # Transcribe (main processing)
            print(f"{self.emoji('🔄', '*')} Transcribing audio...")
            transcribe_start = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(audio, batch_size=16)
            
            transcribe_time = time.time() - transcribe_start
            print(f"{self.emoji('⚡', '>')} Transcription completed in {transcribe_time:.1f}s")
            print("progress=60")
            
            # Language detection
            language = result.get("language", "en")
            print(f"{self.emoji('🌍', '@')} Detected language: {language}")
            
            # Word-level alignment (using preloaded model if available)
            try:
                print(f"{self.emoji('🎯', '>')} Adding word-level timestamps...")
                print("progress=70")
                
                # Use preloaded alignment model if available and matches language
                if (self.align_model is not None and 
                    self.align_metadata is not None and 
                    language == "en"):  # Our preloaded model is for English
                    
                    print(f"{self.emoji('⚡', '>')} Using preloaded alignment model (faster!)")
                    align_start = time.time()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = self.whisperx.align(
                            result["segments"], 
                            self.align_model, 
                            self.align_metadata, 
                            audio, 
                            self.device, 
                            return_char_alignments=False
                        )
                    
                    align_time = time.time() - align_start
                    print(f"{self.emoji('⚡', '>')} Alignment completed in {align_time:.1f}s")
                    
                else:
                    # Load alignment model for detected language
                    print(f"{self.emoji('📥', '>')} Loading alignment model for {language}...")
                    align_start = time.time()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model_a, metadata = self.whisperx.load_align_model(
                            language_code=language, 
                            device=self.device
                        )
                        result = self.whisperx.align(
                            result["segments"], 
                            model_a, 
                            metadata, 
                            audio, 
                            self.device, 
                            return_char_alignments=False
                        )
                    
                    align_time = time.time() - align_start
                    print(f"{self.emoji('⚡', '>')} Alignment completed in {align_time:.1f}s")
                
                print("progress=90")
                    
            except Exception as align_error:
                print(f"{self.emoji('⚠️', '!')} Warning: Word-level alignment failed: {align_error}")
                print(f"{self.emoji('📝', '-')} Proceeding with segment-level timestamps only...")
                result = {"segments": result["segments"], "language": language}
            
            # Format output
            transcript_text = " ".join([
                segment.get("text", "").strip() 
                for segment in result["segments"]
            ]).strip()
            
            output = {
                "transcript": transcript_text,
                "language": result.get("language", language),
                "model_used": self.current_model_name,
                "audio_duration": round(audio_duration, 2),
                "processing_time": round(time.time() - start_time, 2),
                "segments": []
            }
            
            # Process segments
            for segment in result["segments"]:
                segment_data = {
                    "start": round(segment.get("start", 0), 3),
                    "end": round(segment.get("end", 0), 3),
                    "text": segment.get("text", "").strip(),
                    "words": []
                }
                
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        word_data = {
                            "word": word.get("word", ""),
                            "start": round(word.get("start", 0), 3),
                            "end": round(word.get("end", 0), 3),
                            "score": round(word.get("score", 0), 3)
                        }
                        segment_data["words"].append(word_data)
                
                output["segments"].append(segment_data)
            
            total_time = time.time() - start_time
            speed_ratio = audio_duration / total_time if total_time > 0 else 0
            
            print("progress=100")
            print(f"{self.emoji('🎉', '!')} Transcription completed!")
            print(f"{self.emoji('⚡', '>')} Total time: {total_time:.1f}s (Speed: {speed_ratio:.1f}x realtime)")
            print(f"{self.emoji('📊', '#')} Found {len(output['segments'])} segments")
            print("=" * 60)
            print(json.dumps(output, indent=2, ensure_ascii=False))
            print("=" * 60)
            
        except FileNotFoundError:
            print(f"{self.emoji('❌', 'X')} Error: Audio file not found: {audio_path}")
        except PermissionError:
            print(f"{self.emoji('❌', 'X')} Error: Permission denied accessing: {audio_path}")
        except Exception as e:
            print(f"{self.emoji('❌', 'X')} Error transcribing audio: {e}")
            print("\nTroubleshooting:")
            print("- Ensure audio file format is supported")
            print("- Check file is not corrupted or in use")
            print("- Try with a different audio file")
            print("- Restart script if memory issues occur")
    
    def parse_command(self, command):
        """Parse command and extract function name and arguments"""
        command = command.strip()
        
        if command == "exit()":
            return "exit", []
        elif command == "list-models()":
            return "list-models", []
        elif command.startswith("load-model(") and command.endswith(")"):
            arg = command[11:-1].strip()
            return "load-model", [arg]
        elif command.startswith("transcribe-audio(") and command.endswith(")"):
            arg = command[17:-1].strip()
            return "transcribe-audio", [arg]
        else:
            return None, []
    
    def run(self):
        """Main command loop"""
        print(f"{self.emoji('🎙️', '#')} WhisperX Command Line Interface")
        print("=" * 50)
        
        if not self.setup_environment():
            print(f"{self.emoji('❌', 'X')} Failed to setup environment. Exiting.")
            return
        
        print("\nAvailable commands:")
        print("- list-models()                    # Show all available models")
        print("- load-model(model_name)          # Load & optimize model for speed")
        print("- transcribe-audio(audio_path)    # Fast transcription")
        print("- exit()                          # Exit program")
        print("\nPro tip: Use load-model() once, then transcribe multiple files fast!")
        print("Recommended: Start with load-model(tiny.en) for fastest loading")
        print("\nType commands below:")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                cmd_name, args = self.parse_command(command)
                
                if cmd_name == "exit":
                    print(f"{self.emoji('👋', 'Bye!')} Goodbye!")
                    break
                elif cmd_name == "list-models":
                    self.list_models()
                elif cmd_name == "load-model":
                    if args:
                        self.load_model(args[0])
                    else:
                        print(f"{self.emoji('❌', 'X')} Error: load-model requires a model name")
                        print(f"{self.emoji('💡', 'i')} Example: load-model(tiny.en)")
                        print("Use list-models() to see all available models")
                elif cmd_name == "transcribe-audio":
                    if args:
                        self.transcribe_audio(args[0])
                    else:
                        print(f"{self.emoji('❌', 'X')} Error: transcribe-audio requires an audio file path")
                        print(f"{self.emoji('💡', 'i')} Example: transcribe-audio('audio.wav')")
                        print(f"{self.emoji('💡', 'i')} Example: transcribe-audio(C:\\path\\to\\audio.mp3)")
                else:
                    print(f"{self.emoji('❌', 'X')} Invalid command. Available commands:")
                    print("- list-models()")
                    print("- load-model(model_name)")
                    print("- transcribe-audio(audio_path)")
                    print("- exit()")
                    
            except KeyboardInterrupt:
                print(f"\n{self.emoji('👋', 'Bye!')} Goodbye!")
                break
            except Exception as e:
                print(f"{self.emoji('❌', 'X')} Unexpected error: {e}")
                print("Continuing... Type exit() to quit")

if __name__ == "__main__":
    cli = WhisperXCLI()
    cli.run()
