import bpy
import subprocess
import threading
import json
import re
import os
import sys
import time
from pathlib import Path
from bpy.props import StringProperty, EnumProperty, BoolProperty, IntProperty
from bpy.types import Panel, Operator, PropertyGroup

bl_info = {
    "name": "WhisperX FLOW - Advanced Audio Transcription",
    "author": "NeoEmberArts",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > WhisperX and Properties > Tools > WhisperX",
    "description": "Professional audio transcription toolkit using OpenAI's WhisperX with precise word-level timestamps. Create NLA animation strips, video editor subtitles, and export transcripts for animation and video production workflows.",
    "category": "Audio",
    "doc_url": "https://github.com/NeoEmberArt/WhisperXFLOW",
    "tracker_url": "https://github.com/NeoEmberArt/WhisperXFLOW/issues",
    "support": "COMMUNITY",
    "warning": "External dependencies required: WhisperX Python package, PyTorch, and whisperx_runner.py script. ",
    "wiki_url": "https://github.com/NeoEmberArt/WhisperXFLOW/wiki",
    "tags": ["Audio", "Transcription", "Animation", "Video", "Subtitles", "NLA", "AI", "Speech-to-Text"],
}


# Global variables
whisperx_process = None
process_thread = None
ui_update_timer = None
last_ui_update = 0
UI_UPDATE_INTERVAL = 0.5  # Update UI every 0.5 seconds during processing

# Constants
BUFFER_OBJECT_NAME = "WhisperX_Transcript_Buffer"
VSE_STRIP_PREFIX = "WhisperX_Sub_"

# States for UI flow control
STATE_INITIAL = 0      # Initial state - service not running
STATE_RUNNING = 1      # Service running - no model loaded
STATE_MODEL_READY = 2  # Model loaded - ready for transcription
STATE_PROCESSING = 3   # Currently processing (loading model or transcribing)
STATE_TRANSCRIBED = 4  # Transcription complete - show output options

class WhisperXProperties(PropertyGroup):
    # UI State
    ui_state: IntProperty(
        name="UI State",
        default=STATE_INITIAL
    )
    
    # Process control
    process_running: BoolProperty(
        name="Process Running",
        default=False
    )
    
    # Model selection
    available_models: EnumProperty(
        name="Model",
        description="Select a model to load",
        items=[
            ("tiny", "Tiny (39 MB) - Fastest", "Tiny model - fastest but least accurate"),
            ("tiny.en", "Tiny.en (39 MB) - English only", "Tiny English-only model"),
            ("base", "Base (74 MB) - Balanced", "Base model - good balance of speed and accuracy"),
            ("base.en", "Base.en (74 MB) - English only", "Base English-only model"),
            ("small", "Small (244 MB) - Accurate", "Small model - better accuracy"),
            ("small.en", "Small.en (244 MB) - English only", "Small English-only model"),
            ("medium", "Medium (769 MB) - Very accurate", "Medium model - high accuracy"),
            ("medium.en", "Medium.en (769 MB) - English only", "Medium English-only model"),
            ("large-v3", "Large-v3 (1.5 GB) - Most accurate", "Large model v3 - best accuracy"),
        ],
        default="tiny.en"
    )
    
    # Audio file path
    audio_file_path: StringProperty(
        name="Audio File",
        description="Select an audio file to transcribe",
        default="",
        subtype='FILE_PATH'
    )
    
    # Script path
    script_path: StringProperty(
        name="Script Path",
        description="Path to whisperx_runner.py script",
        default="",
        subtype='FILE_PATH'
    )
    
    # Status and logs
    status_message: StringProperty(
        name="Status",
        default="Ready to start"
    )
    
    loaded_model: StringProperty(
        name="Loaded Model",
        default="None"
    )
    
    # Output logs
    process_log: StringProperty(
        name="Process Log",
        default=""
    )
    
    transcription_output: StringProperty(
        name="Transcription Output",
        default=""
    )
    
    # Show advanced options
    show_advanced: BoolProperty(
        name="Show Advanced Options",
        default=False
    )
    
    # Last update timestamp
    last_update_time: IntProperty(
        name="Last Update Time",
        default=0
    )
    
    # NLA strip settings
    nla_buffer_name: StringProperty(
        name="NLA Buffer Name",
        default=BUFFER_OBJECT_NAME
    )
    
    # Video Editor settings
    vse_text_size: IntProperty(
        name="Text Size",
        description="Size of text in the Video Sequencer",
        default=70,
        min=10,
        max=200
    )
    
    vse_text_position: EnumProperty(
        name="Text Position",
        description="Position of text in the Video Sequencer",
        items=[
            ("BOTTOM", "Bottom", "Position text at the bottom"),
            ("CENTER", "Center", "Position text at the center"),
            ("TOP", "Top", "Position text at the top"),
        ],
        default="BOTTOM"
    )
    
    # Word vs Segment mode for VSE
    vse_use_words: BoolProperty(
        name="Use Words (not Segments)",
        description="Create a strip for each word instead of each segment",
        default=True
    )
    
    # Add audio to scene option
    add_audio_to_scene: BoolProperty(
        name="Add Audio to Scene",
        description="Add the audio file to the scene after transcription",
        default=False
    )

# Helper function to force UI refresh
def force_ui_update():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()

# Timer function to update UI during processing
def ui_update_timer_callback():
    global last_ui_update
    
    # Update UI at regular intervals
    current_time = time.time()
    if current_time - last_ui_update >= UI_UPDATE_INTERVAL:
        force_ui_update()
        last_ui_update = current_time
    
    # Continue timer if process is running
    if bpy.context.scene.whisperx_props.process_running:
        return UI_UPDATE_INTERVAL
    else:
        # One final update when process ends
        force_ui_update()
        return None

# Helper functions for thread-safe UI updates
def add_to_log(context, line):
    def _update():
        props = context.scene.whisperx_props
        props.process_log += line + "\n"
        
        # Keep log manageable (last 100 lines)
        lines = props.process_log.split('\n')
        if len(lines) > 100:
            props.process_log = '\n'.join(lines[-100:])
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

def update_loaded_model(context, model_name):
    def _update():
        props = context.scene.whisperx_props
        props.loaded_model = model_name
        props.status_message = f"Model '{model_name}' loaded"
        props.ui_state = STATE_MODEL_READY
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

def update_transcription_output(context, json_text):
    def _update():
        props = context.scene.whisperx_props
        props.transcription_output = json_text
        props.ui_state = STATE_TRANSCRIBED  # Set to transcribed state to show output options
        props.status_message = "Transcription complete"
        
        # Add audio to scene if option is enabled
        if props.add_audio_to_scene and props.audio_file_path:
            try:
                add_audio_to_scene(context, props.audio_file_path)
            except Exception as e:
                print(f"Error adding audio to scene: {e}")
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

def update_status(context, message):
    def _update():
        props = context.scene.whisperx_props
        props.status_message = message
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

def update_ui_state(context, state):
    def _update():
        props = context.scene.whisperx_props
        props.ui_state = state
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

def process_ended(context, exit_code=None):
    def _update():
        global whisperx_process, process_thread, ui_update_timer
        
        props = context.scene.whisperx_props
        props.process_running = False
        props.ui_state = STATE_INITIAL
        props.loaded_model = "None"
        
        if exit_code is not None and exit_code != 0:
            props.status_message = f"Service stopped with error (code: {exit_code})"
            props.process_log += f"Process exited with error code: {exit_code}\n"
        else:
            props.status_message = "Service stopped"
            props.process_log += "Process ended normally\n"
        
        # Clean up global variables
        whisperx_process = None
        process_thread = None
        
        # Stop UI update timer
        if ui_update_timer is not None and bpy.app.timers.is_registered(ui_update_timer_callback):
            try:
                bpy.app.timers.unregister(ui_update_timer_callback)
            except:
                pass
        
        # Force UI update
        props.last_update_time = int(time.time())
        force_ui_update()
        return None
    
    bpy.app.timers.register(_update)

# Thread function to read process output
def read_process_output(process, context):
    props = context.scene.whisperx_props
    json_capture = False
    json_lines = []
    
    while True:
        # Check if process has ended
        if process.poll() is not None:
            exit_code = process.poll()
            add_to_log(context, f"Process exited with code {exit_code}")
            process_ended(context, exit_code)
            break
        
        # Read output line
        try:
            line = process.stdout.readline()
            if not line:
                # No more output but process still running
                time.sleep(0.1)
                continue
                
            # Handle encoding issues
            try:
                line = line.strip()
            except UnicodeDecodeError:
                line = "Error decoding output line (Unicode error)"
            
            # Detect JSON output start/end
            if line.startswith('============================================================'):
                if json_capture:
                    # End of JSON, process it
                    json_text = '\n'.join(json_lines)
                    update_transcription_output(context, json_text)
                    json_capture = False
                    json_lines = []
                else:
                    # Start of JSON
                    json_capture = True
            elif json_capture:
                json_lines.append(line)
            else:
                # Regular log update
                add_to_log(context, line)
            
            # Parse status messages
            if "Model" in line and "loaded" in line:
                model_match = re.search(r"Model '([^']+)'", line)
                if model_match:
                    model_name = model_match.group(1)
                    update_loaded_model(context, model_name)
            
            # Update status based on key phrases
            if "Error:" in line:
                update_status(context, f"Error: {line.split('Error:')[1].strip()}")
            elif "Starting" in line or "Setting up" in line:
                update_status(context, "Starting service...")
            elif "Environment setup complete" in line:
                update_status(context, "Service running")
                update_ui_state(context, STATE_RUNNING)
            elif "Transcribing" in line:
                update_status(context, "Transcribing audio...")
            elif "Downloading" in line or "Loading model" in line:
                update_status(context, "Loading model...")
            elif "Transcription completed" in line:
                update_status(context, "Processing transcription...")
            
        except Exception as e:
            add_to_log(context, f"Error reading output: {str(e)}")
            time.sleep(0.1)

def find_whisperx_script():
    """Find the whisperx_runner.py script in common locations"""
    # First check the addon directory
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    possible_paths = [
        os.path.join(addon_dir, "whisperx_runner.py"),
        os.path.join(os.getcwd(), "whisperx_runner.py"),
        os.path.join(os.path.expanduser("~"), "whisperx_runner.py"),
        os.path.join(os.path.expanduser("~"), "Downloads", "whisperx_runner.py"),
    ]
    
    # Add common project directories
    project_dirs = [
        "C:/Users/obsid/Downloads/Website/Python Projects/VisemeNueronV2",
        os.path.join(os.path.expanduser("~"), "Documents"),
        os.path.join(os.path.expanduser("~"), "Projects"),
    ]
    
    for project_dir in project_dirs:
        if os.path.exists(project_dir):
            possible_paths.append(os.path.join(project_dir, "whisperx_runner.py"))
    
    # Check all paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

# NLA Helper Functions
def get_or_create_buffer_object(buffer_name):
    """Get or create a buffer object for NLA strips"""
    obj = bpy.data.objects.get(buffer_name)
    if obj is None:
        obj = bpy.data.objects.new(buffer_name, None)
        bpy.context.collection.objects.link(obj)
    if obj.animation_data is None:
        obj.animation_data_create()
    return obj

def delete_buffer_object(buffer_name):
    """Delete the buffer object used for NLA strips"""
    obj = bpy.data.objects.get(buffer_name)
    if obj:
        if obj.animation_data and obj.animation_data.nla_tracks:
            for track in obj.animation_data.nla_tracks:
                obj.animation_data.nla_tracks.remove(track)
        bpy.data.objects.remove(obj, do_unlink=True)
        return True
    return False

# VSE Helper Functions
def delete_vse_subtitle_strips():
    """Delete all VSE text strips created by this addon"""
    scene = bpy.context.scene
    seq = scene.sequence_editor
    
    if not seq:
        return 0
    
    to_remove = [s for s in seq.sequences_all if s.name.startswith(VSE_STRIP_PREFIX)]
    for s in to_remove:
        seq.sequences.remove(s)
    return len(to_remove)

# Audio Helper Functions
def add_audio_to_scene(context, audio_path):
    """Add audio file to the scene as a speaker object"""
    # Create a new speaker object
    speaker_data = bpy.data.speakers.new(name="WhisperX_Audio")
    speaker_obj = bpy.data.objects.new("WhisperX_Audio", speaker_data)
    
    # Link the speaker to the scene
    context.collection.objects.link(speaker_obj)
    
    # Load the audio file
    sound = bpy.data.sounds.load(audio_path)
    
    # Assign the sound to the speaker
    speaker_data.sound = sound
    
    # Position the speaker at the origin
    speaker_obj.location = (0, 0, 0)
    
    # Select the speaker
    for obj in context.selected_objects:
        obj.select_set(False)
    speaker_obj.select_set(True)
    context.view_layer.objects.active = speaker_obj
    
    return speaker_obj

class WHISPERX_OT_locate_script(Operator):
    bl_idname = "whisperx.locate_script"
    bl_label = "Locate Script"
    bl_description = "Find the whisperx_runner.py script"
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        default="whisperx_runner.py"
    )
    
    def execute(self, context):
        props = context.scene.whisperx_props
        props.script_path = self.filepath
        props.status_message = f"Script path set"
        force_ui_update()
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class WHISPERX_OT_start_service(Operator):
    bl_idname = "whisperx.start_service"
    bl_label = "Start Service"
    bl_description = "Start the WhisperX FLOW service"
    
    def execute(self, context):
        global whisperx_process, process_thread, ui_update_timer, last_ui_update
        props = context.scene.whisperx_props
        
        if props.process_running:
            self.report({'WARNING'}, "Service is already running")
            return {'CANCELLED'}
        
        try:
            # Find the script
            whisperx_script = props.script_path
            if not whisperx_script or not os.path.exists(whisperx_script):
                whisperx_script = find_whisperx_script()
                if whisperx_script:
                    props.script_path = whisperx_script
                    props.status_message = f"Found script"
                else:
                    props.status_message = "Error: Script not found"
                    self.report({'ERROR'}, "whisperx_runner.py not found. Please use 'Locate Script' button.")
                    return {'CANCELLED'}
            
            # Get Python executable
            python_exe = sys.executable
            
            # Set environment variables for encoding
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            # Get the directory of the script
            script_dir = os.path.dirname(os.path.abspath(whisperx_script))
            
            # Start the subprocess with encoding settings
            whisperx_process = subprocess.Popen(
                [python_exe, whisperx_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                encoding='utf-8',
                errors='replace',  # Handle encoding errors gracefully
                cwd=script_dir  # Set working directory to script directory
            )
            
            # Start reading output in a separate thread
            process_thread = threading.Thread(
                target=read_process_output, 
                args=(whisperx_process, context)
            )
            process_thread.daemon = True
            process_thread.start()
            
            # Start UI update timer
            last_ui_update = time.time()
            if ui_update_timer is None or not bpy.app.timers.is_registered(ui_update_timer_callback):
                bpy.app.timers.register(ui_update_timer_callback)
            
            props.process_running = True
            props.status_message = "Starting service..."
            props.process_log = "Initializing WhisperX service...\n"
            props.ui_state = STATE_PROCESSING
            
            # Force immediate UI update
            force_ui_update()
            
            return {'FINISHED'}
            
        except Exception as e:
            error_msg = f"Error starting service: {str(e)}"
            props.status_message = error_msg
            self.report({'ERROR'}, error_msg)
            force_ui_update()
            return {'CANCELLED'}

class WHISPERX_OT_stop_service(Operator):
    bl_idname = "whisperx.stop_service"
    bl_label = "Stop Service"
    bl_description = "Stop the WhisperX FLOW service"
    
    def execute(self, context):
        global whisperx_process, process_thread, ui_update_timer
        props = context.scene.whisperx_props
        
        if not props.process_running:
            self.report({'WARNING'}, "No service is running")
            return {'CANCELLED'}
        
        try:
            # First, update the UI state immediately
            props.process_running = False
            props.status_message = "Stopping service..."
            props.ui_state = STATE_INITIAL
            props.loaded_model = "None"
            
            # Stop the UI update timer
            if ui_update_timer is not None and bpy.app.timers.is_registered(ui_update_timer_callback):
                try:
                    bpy.app.timers.unregister(ui_update_timer_callback)
                except:
                    pass
            
            # Handle process termination
            if whisperx_process and whisperx_process.poll() is None:
                add_to_log(context, "Stopping WhisperX service...")
                
                # Method 1: Try graceful shutdown with exit command
                try:
                    if whisperx_process.stdin and not whisperx_process.stdin.closed:
                        whisperx_process.stdin.write("exit()\n")
                        whisperx_process.stdin.flush()
                        whisperx_process.stdin.close()
                        
                        # Wait up to 3 seconds for graceful exit
                        for _ in range(30):  # 30 * 0.1 = 3 seconds
                            if whisperx_process.poll() is not None:
                                add_to_log(context, "Service stopped gracefully")
                                break
                            time.sleep(0.1)
                except Exception as e:
                    add_to_log(context, f"Could not send exit command: {e}")
                
                # Method 2: If still running, try terminate
                if whisperx_process.poll() is None:
                    try:
                        add_to_log(context, "Terminating process...")
                        whisperx_process.terminate()
                        
                        # Wait up to 2 seconds for termination
                        for _ in range(20):  # 20 * 0.1 = 2 seconds
                            if whisperx_process.poll() is not None:
                                add_to_log(context, "Service terminated")
                                break
                            time.sleep(0.1)
                    except Exception as e:
                        add_to_log(context, f"Could not terminate process: {e}")
                
                # Method 3: If still running, force kill
                if whisperx_process.poll() is None:
                    try:
                        add_to_log(context, "Force killing process...")
                        whisperx_process.kill()
                        time.sleep(0.5)
                        if whisperx_process.poll() is not None:
                            add_to_log(context, "Service force killed")
                        else:
                            add_to_log(context, "Warning: Could not kill process")
                    except Exception as e:
                        add_to_log(context, f"Could not kill process: {e}")
            
            # Clean up global variables
            whisperx_process = None
            
            # Handle thread cleanup
            if process_thread and process_thread.is_alive():
                # The thread should exit naturally when the process ends
                # We don't force-kill threads as it's dangerous
                add_to_log(context, "Waiting for process thread to finish...")
                
                # Give the thread a moment to detect the process has ended
                def cleanup_thread():
                    if process_thread and process_thread.is_alive():
                        # Thread is still alive after process ended - this is unusual
                        add_to_log(context, "Process thread cleanup completed")
                    return None
                
                # Schedule cleanup check after a short delay
                bpy.app.timers.register(cleanup_thread, first_interval=1.0)
            
            process_thread = None
            
            # Final status update
            props.status_message = "Service stopped"
            props.process_log += "=== Service stopped by user ===\n"
            
            # Force final UI update
            force_ui_update()
            
            self.report({'INFO'}, "Service stopped successfully")
            return {'FINISHED'}
            
        except Exception as e:
            error_msg = f"Error stopping service: {str(e)}"
            props.status_message = error_msg
            add_to_log(context, error_msg)
            self.report({'ERROR'}, error_msg)
            
            # Even if there was an error, reset the state
            props.process_running = False
            props.ui_state = STATE_INITIAL
            props.loaded_model = "None"
            whisperx_process = None
            process_thread = None
            
            force_ui_update()
            return {'CANCELLED'}


class WHISPERX_OT_load_model(Operator):
    bl_idname = "whisperx.load_model"
    bl_label = "Load Model"
    bl_description = "Load the selected model"
    
    def execute(self, context):
        global whisperx_process
        props = context.scene.whisperx_props
        
        if not props.process_running:
            self.report({'WARNING'}, "Service is not running. Start it first.")
            return {'CANCELLED'}
        
        try:
            if whisperx_process and whisperx_process.poll() is None:
                model_name = props.available_models
                command = f"load-model({model_name})\n"
                whisperx_process.stdin.write(command)
                whisperx_process.stdin.flush()
                
                props.status_message = f"Loading model..."
                props.process_log += f"Loading model: {model_name}...\n"
                props.ui_state = STATE_PROCESSING
                
                # Force UI update
                force_ui_update()
            else:
                props.status_message = "Error: Service not responding"
                props.process_running = False
                props.ui_state = STATE_INITIAL
                force_ui_update()
                
        except Exception as e:
            self.report({'ERROR'}, f"Error loading model: {str(e)}")
            force_ui_update()
            return {'CANCELLED'}
        
        return {'FINISHED'}

class WHISPERX_OT_transcribe_audio(Operator):
    bl_idname = "whisperx.transcribe_audio"
    bl_label = "Transcribe"
    bl_description = "Transcribe the selected audio file"
    
    def execute(self, context):
        global whisperx_process
        props = context.scene.whisperx_props
        
        if not props.process_running:
            self.report({'WARNING'}, "Service is not running. Start it first.")
            return {'CANCELLED'}
        
        if not props.audio_file_path:
            self.report({'WARNING'}, "Please select an audio file first.")
            return {'CANCELLED'}
        
        if props.loaded_model == "None":
            self.report({'WARNING'}, "Please load a model first.")
            return {'CANCELLED'}
        
        try:
            if whisperx_process and whisperx_process.poll() is None:
                # Clear previous transcription output
                props.transcription_output = ""
                
                audio_path = bpy.path.abspath(props.audio_file_path)
                # Replace backslashes with forward slashes for consistent handling
                audio_path = audio_path.replace("\\", "/")
                command = f'transcribe-audio("{audio_path}")\n'
                
                whisperx_process.stdin.write(command)
                whisperx_process.stdin.flush()
                
                props.status_message = "Transcribing audio..."
                props.process_log += f"Transcribing: {os.path.basename(audio_path)}...\n"
                props.ui_state = STATE_PROCESSING
                
                # Force UI update
                force_ui_update()
            else:
                props.status_message = "Error: Service not responding"
                props.process_running = False
                props.ui_state = STATE_INITIAL
                force_ui_update()
                
        except Exception as e:
            self.report({'ERROR'}, f"Error transcribing audio: {str(e)}")
            force_ui_update()
            return {'CANCELLED'}
        
        return {'FINISHED'}

class WHISPERX_OT_try_again(Operator):
    bl_idname = "whisperx.try_again"
    bl_label = "Try Again"
    bl_description = "Select a new audio file and transcribe again"
    
    filepath: StringProperty(
        subtype='FILE_PATH',
        default=""
    )
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        # Set the new audio path
        if self.filepath:
            props.audio_file_path = self.filepath
        
        # Return to model ready state
        props.ui_state = STATE_MODEL_READY
        props.status_message = f"Model '{props.loaded_model}' loaded"
        
        # Force UI update
        force_ui_update()
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class WHISPERX_OT_clear_logs(Operator):
    bl_idname = "whisperx.clear_logs"
    bl_label = "Clear Logs"
    bl_description = "Clear the process logs"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        props.process_log = ""
        force_ui_update()
        return {'FINISHED'}

class WHISPERX_OT_toggle_advanced(Operator):
    bl_idname = "whisperx.toggle_advanced"
    bl_label = "Advanced Options"
    bl_description = "Show or hide advanced options"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        props.show_advanced = not props.show_advanced
        force_ui_update()
        return {'FINISHED'}

class WHISPERX_OT_send_to_nla(Operator):
    bl_idname = "whisperx.send_to_nla"
    bl_label = "Send to NLA"
    bl_description = "Create NLA strips for each word in the transcription"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        try:
            # Parse the transcription output
            data = json.loads(props.transcription_output)
            
            # Extract words from segments
            words = []
            for segment in data.get('segments', []):
                words.extend(segment.get('words', []))
            
            if not words:
                self.report({'ERROR'}, "No words found in transcription data")
                return {'CANCELLED'}
            
            # Get or create buffer object
            buffer_name = props.nla_buffer_name
            obj = get_or_create_buffer_object(buffer_name)
            
            # Clear existing NLA tracks
            if obj.animation_data.nla_tracks:
                for track in obj.animation_data.nla_tracks:
                    obj.animation_data.nla_tracks.remove(track)
            
            # Create a new NLA track
            track = obj.animation_data.nla_tracks.new()
            track.name = "Transcript"
            
            # Get scene FPS for frame conversion
            fps = context.scene.render.fps
            
            # Process each word
            for i, word_data in enumerate(words):
                word = word_data.get('word', '').strip()
                if not word:
                    continue
                
                start_time = word_data.get('start', 0)
                end_time = word_data.get('end', start_time + 0.5)
                
                # Convert to frames
                start_frame = int(start_time * fps) + 1  # +1 to avoid frame 0
                end_frame = int(end_time * fps) + 1
                
                # Ensure minimum strip length
                if end_frame <= start_frame:
                    end_frame = start_frame + 1
                
                # Create a new action for the word
                action_name = f"Word_{i:03d}_{word}"
                action = bpy.data.actions.new(action_name)
                
                # Dummy fcurve to make the action valid
                data_path = '["dummy_prop"]'
                fcurve = action.fcurves.new(data_path=data_path)
                fcurve.keyframe_points.add(2)
                fcurve.keyframe_points[0].co = (1, 1)
                fcurve.keyframe_points[1].co = (8, 0)
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
                
                # Create NLA strip
                strip = track.strips.new(name=word, start=start_frame, action=action)
                strip.frame_end = end_frame
                
                # Set strip name based on confidence score
                confidence = word_data.get('score', 0)
                if confidence > 0.8:
                    # For high confidence, make the strip name uppercase
                    strip.name = word.upper()
                elif confidence < 0.5:
                    # For low confidence, add a question mark
                    strip.name = word + "?"
            
            # Clear any active action
            obj.animation_data.action = None
            
            self.report({'INFO'}, f"Created {len(words)} NLA strips for transcript")
            return {'FINISHED'}
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            self.report({'ERROR'}, f"Error processing transcription data: {str(e)}")
            return {'CANCELLED'}

class WHISPERX_OT_remove_from_nla(Operator):
    bl_idname = "whisperx.remove_from_nla"
    bl_label = "Remove from NLA"
    bl_description = "Remove NLA strips created from transcription"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        try:
            buffer_name = props.nla_buffer_name
            removed = delete_buffer_object(buffer_name)
            
            if removed:
                self.report({'INFO'}, "Removed transcript buffer object and strips")
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "Transcript buffer object not found")
                return {'CANCELLED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error removing NLA data: {str(e)}")
            return {'CANCELLED'}

class WHISPERX_OT_send_to_vse(Operator):
    bl_idname = "whisperx.send_to_vse"
    bl_label = "Send to Video Editor"
    bl_description = "Create text strips in the Video Sequencer for each word or segment"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        try:
            # Parse the transcription output
            data = json.loads(props.transcription_output)
            
            # Make sure we have a sequence editor
            if not context.scene.sequence_editor:
                context.scene.sequence_editor_create()
            
            # Clear existing subtitle strips
            removed = delete_vse_subtitle_strips()
            if removed > 0:
                print(f"Removed {removed} existing subtitle strips")
            
            # Find a single channel for all text strips
            text_channel = 2  # Start at channel 2 to leave channel 1 for audio
            # Find the highest used channel and add 1
            for strip in context.scene.sequence_editor.sequences_all:
                if strip.channel >= text_channel:
                    text_channel = strip.channel + 1
            
            # Get scene FPS for frame conversion
            fps = context.scene.render.fps
            
            # Determine what to process based on user preference
            if props.vse_use_words:
                # Process each word
                items = []
                for segment in data.get('segments', []):
                    for word in segment.get('words', []):
                        items.append({
                            'text': word.get('word', '').strip(),
                            'start': word.get('start', 0),
                            'end': word.get('end', 0)
                        })
            else:
                # Process each segment
                items = []
                for segment in data.get('segments', []):
                    items.append({
                        'text': segment.get('text', '').strip(),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0)
                    })
            
            # Create text strips - ALL ON THE SAME CHANNEL
            created_count = 0
            for i, item in enumerate(items):
                text = item['text']
                if not text:
                    continue
                
                start_time = item['start']
                end_time = item['end']
                
                # Convert times to frames
                start_frame = int(start_time * fps) + 1  # +1 to avoid frame 0
                end_frame = int(end_time * fps) + 1
                
                # Ensure minimum strip length
                if end_frame <= start_frame:
                    end_frame = start_frame + 1
                
                # Create text strip - ALL USE THE SAME CHANNEL
                strip_name = f"{VSE_STRIP_PREFIX}{i:03d}_{text[:10]}"  # Include part of text in name
                text_strip = context.scene.sequence_editor.sequences.new_effect(
                    name=strip_name,
                    type='TEXT',
                    channel=text_channel,  # Same channel for all text strips
                    frame_start=start_frame,
                    frame_end=end_frame
                )
                
                # Set text properties
                text_strip.text = text
                text_strip.font_size = props.vse_text_size
                
                # Set position based on user preference
                if props.vse_text_position == "BOTTOM":
                    text_strip.location = [0.5, 0.1]  # Bottom center
                elif props.vse_text_position == "CENTER":
                    text_strip.location = [0.5, 0.5]  # Center
                else:  # TOP
                    text_strip.location = [0.5, 0.9]  # Top center
                
                # Set color (white with full alpha)
                text_strip.color = (1.0, 1.0, 1.0, 1.0)
                
                # Optional: Add shadow for better readability
                if hasattr(text_strip, 'use_shadow'):
                    text_strip.use_shadow = True
                    text_strip.shadow_color = (0.0, 0.0, 0.0, 0.8)
                
                # Optional: Set wrap width if available
                if hasattr(text_strip, 'wrap_width'):
                    text_strip.wrap_width = 0.8
                
                created_count += 1
            
            # Add audio to VSE on a separate channel (channel 1)
            if props.audio_file_path:
                audio_path = bpy.path.abspath(props.audio_file_path)
                if os.path.exists(audio_path):
                    audio_name = os.path.basename(audio_path)
                    
                    # Check if audio strip already exists
                    audio_exists = False
                    for strip in context.scene.sequence_editor.sequences_all:
                        if (strip.type == 'SOUND' and 
                            hasattr(strip, 'sound') and 
                            strip.sound and 
                            (strip.sound.filepath.endswith(audio_name) or 
                             strip.name.startswith("WhisperX_Audio"))):
                            audio_exists = True
                            break
                    
                    # Add audio if it doesn't exist - on channel 1
                    if not audio_exists:
                        try:
                            # Use channel 1 for audio
                            audio_channel = 1
                            
                            # Create the sound strip
                            sound_strip = context.scene.sequence_editor.sequences.new_sound(
                                name=f"WhisperX_Audio_{audio_name}",
                                filepath=audio_path,
                                channel=audio_channel,
                                frame_start=1
                            )
                            print(f"Added audio strip on channel {audio_channel}: {sound_strip.name}")
                            
                        except Exception as audio_error:
                            print(f"Could not add audio to VSE: {audio_error}")
                            # Don't fail the whole operation if audio can't be added
            
            self.report({'INFO'}, f"Created {created_count} text strips on channel {text_channel}")
            
            # Switch to Video Editing workspace if it exists
            try:
                for workspace in bpy.data.workspaces:
                    if workspace.name == "Video Editing":
                        context.window.workspace = workspace
                        break
            except:
                pass  # Don't fail if workspace switching doesn't work
            
            return {'FINISHED'}
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            self.report({'ERROR'}, f"Error processing transcription data: {str(e)}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Unexpected error: {str(e)}")
            return {'CANCELLED'}

class WHISPERX_OT_remove_from_vse(Operator):
    bl_idname = "whisperx.remove_from_vse"
    bl_label = "Remove from Video Editor"
    bl_description = "Remove text strips created from transcription"
    
    def execute(self, context):
        try:
            removed = delete_vse_subtitle_strips()
            
            if removed > 0:
                self.report({'INFO'}, f"Removed {removed} text strips from Video Sequencer")
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "No text strips found to remove")
                return {'CANCELLED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error removing text strips: {str(e)}")
            return {'CANCELLED'}

class WHISPERX_OT_copy_transcript(Operator):
    bl_idname = "whisperx.copy_transcript"
    bl_label = "Copy Transcript"
    bl_description = "Copy the transcript text to clipboard"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        try:
            data = json.loads(props.transcription_output)
            transcript = data.get('transcript', '')
            
            if transcript:
                context.window_manager.clipboard = transcript
                self.report({'INFO'}, "Transcript copied to clipboard")
            else:
                self.report({'WARNING'}, "No transcript text to copy")
                
        except (json.JSONDecodeError, AttributeError):
            self.report({'ERROR'}, "Could not parse transcript data")
            return {'CANCELLED'}
            
        return {'FINISHED'}

class WHISPERX_OT_add_audio_to_scene(Operator):
    bl_idname = "whisperx.add_audio_to_scene"
    bl_label = "Add Audio to Scene"
    bl_description = "Add the audio file as a speaker object in the scene"
    
    def execute(self, context):
        props = context.scene.whisperx_props
        
        if not props.audio_file_path:
            self.report({'WARNING'}, "No audio file selected")
            return {'CANCELLED'}
        
        try:
            audio_path = bpy.path.abspath(props.audio_file_path)
            speaker_obj = add_audio_to_scene(context, audio_path)
            
            self.report({'INFO'}, f"Added audio as speaker: {speaker_obj.name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error adding audio to scene: {str(e)}")
            return {'CANCELLED'}

class WHISPERX_PT_main_panel(Panel):
    bl_label = "WhisperX FLOW"
    bl_idname = "WHISPERX_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'WhisperX'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.whisperx_props
        
        # Status indicator
        status_box = layout.box()
        row = status_box.row()
        
        # Status icon based on state
        if props.ui_state == STATE_INITIAL:
            row.label(text="", icon='RADIOBUT_OFF')
            status_text = "Service not running"
        elif props.ui_state == STATE_PROCESSING:
            row.label(text="", icon='SORTTIME')
            status_text = props.status_message
        elif props.ui_state == STATE_RUNNING:
            row.label(text="", icon='CHECKMARK')
            status_text = "Service running"
        elif props.ui_state == STATE_MODEL_READY:
            row.label(text="", icon='CHECKMARK')
            status_text = f"Model: {props.loaded_model}"
        elif props.ui_state == STATE_TRANSCRIBED:
            row.label(text="", icon='CHECKMARK')
            status_text = f"Transcription complete"
        else:
            row.label(text="", icon='QUESTION')
            status_text = props.status_message
            
        row.label(text=status_text)
        
        # Main action buttons based on state
        if props.ui_state == STATE_INITIAL:
            # Service not running - show start button
            row = layout.row(align=True)
            row.scale_y = 1.5
            row.operator("whisperx.start_service", icon='PLAY')
            
        elif props.ui_state == STATE_PROCESSING:
            # Processing - show status and stop button
            row = layout.row()
            row.label(text=props.status_message)
            row = layout.row()
            row.operator("whisperx.stop_service", icon='CANCEL')
            
        elif props.ui_state == STATE_RUNNING:
            # Service running but no model loaded
            box = layout.box()
            box.label(text="1. Select Model", icon='PRESET')
            
            # Model selection dropdown
            row = box.row()
            row.prop(props, "available_models", text="")
            
            # Load model button
            row = box.row()
            row.scale_y = 1.2
            row.operator("whisperx.load_model", icon='IMPORT')
            
            # Stop service button
            row = layout.row()
            row.operator("whisperx.stop_service", icon='X')
            
        elif props.ui_state == STATE_MODEL_READY:
            # Model loaded - show transcription options
            box = layout.box()
            box.label(text="1. Select Model", icon='CHECKMARK')
            row = box.row()
            row.label(text=f"Using: {props.loaded_model}")
            
            # Change model option
            row = box.row()
            row.prop(props, "available_models", text="")
            row.operator("whisperx.load_model", text="Change")
            
            # Audio selection
            box = layout.box()
            box.label(text="2. Select Audio", icon='SOUND')
            box.prop(props, "audio_file_path", text="")
            
            # Add audio to scene option
            row = box.row()
            row.prop(props, "add_audio_to_scene", text="Add Audio to Scene")
            
            # Transcribe button
            row = layout.row(align=True)
            row.scale_y = 1.5
            row.enabled = bool(props.audio_file_path)
            row.operator("whisperx.transcribe_audio", icon='REC')
            
            # Stop service button
            row = layout.row()
            row.operator("whisperx.stop_service", icon='X')
            
        elif props.ui_state == STATE_TRANSCRIBED:
            # Transcription complete - show output options
            box = layout.box()
            box.label(text="1. Select Model", icon='CHECKMARK')
            row = box.row()
            row.label(text=f"Using: {props.loaded_model}")
            
            # Audio selection
            box = layout.box()
            box.label(text="2. Select Audio", icon='CHECKMARK')
            row = box.row()
            row.label(text=f"File: {os.path.basename(props.audio_file_path)}")
            
            # Add audio to scene button
            row = box.row()
            row.operator("whisperx.add_audio_to_scene", icon='SPEAKER')
            
            # Output options
            box = layout.box()
            box.label(text="3. Output Options", icon='OUTPUT')
            
            # NLA output
            row = box.row()
            row.label(text="NLA Animation:")
            row = box.row(align=True)
            row.operator("whisperx.send_to_nla", icon='NLA')
            row.operator("whisperx.remove_from_nla", icon='X', text="")
            
            # Video Editor output
            row = box.row()
            row.label(text="Video Editor:")
            row = box.row(align=True)
            row.operator("whisperx.send_to_vse", icon='SEQUENCE')
            row.operator("whisperx.remove_from_vse", icon='X', text="")
            
            # VSE options
            row = box.row()
            row.prop(props, "vse_use_words", text="Use Words (not Segments)")
            
            # Copy transcript button
            row = box.row()
            row.operator("whisperx.copy_transcript", icon='COPYDOWN')
            
            # Try again button
            row = layout.row()
            row.scale_y = 1.2
            row.operator("whisperx.try_again", icon='FILE_REFRESH', text="Try Again")
            
            # Stop service button
            row = layout.row()
            row.operator("whisperx.stop_service", icon='X')
        
        # Advanced options toggle
        row = layout.row()
        if props.show_advanced:
            row.operator("whisperx.toggle_advanced", icon='TRIA_DOWN', text="Hide Advanced Options")
        else:
            row.operator("whisperx.toggle_advanced", icon='TRIA_RIGHT', text="Show Advanced Options")
        
        # Advanced options
        if props.show_advanced:
            box = layout.box()
            box.label(text="Advanced Options", icon='SETTINGS')
            
            # Script path
            row = box.row()
            row.label(text="Script Path:")
            row = box.row()
            row.prop(props, "script_path", text="")
            row.operator("whisperx.locate_script", text="", icon='FILE_FOLDER')
            
            # NLA settings
            row = box.row()
            row.label(text="NLA Settings:")
            row = box.row()
            row.prop(props, "nla_buffer_name", text="Buffer Name")
            
            # Video Editor settings
            row = box.row()
            row.label(text="Video Editor Settings:")
            row = box.row()
            row.prop(props, "vse_text_size", text="Text Size")
            row = box.row()
            row.prop(props, "vse_text_position", text="Position")
            
            # Clear logs button
            row = box.row()
            row.operator("whisperx.clear_logs", icon='TRASH')

class WHISPERX_PT_logs_panel(Panel):
    bl_label = "Process Logs"
    bl_idname = "WHISPERX_PT_logs_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'WhisperX'
    bl_parent_id = "WHISPERX_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.whisperx_props
        
        # Process log
        box = layout.box()
        
        # Show last few lines of log
        if props.process_log:
            lines = props.process_log.split('\n')[-15:]  # Last 15 lines
            for line in lines:
                if line.strip():
                    row = box.row()
                    row.scale_y = 0.8
                    row.label(text=line[:80])  # Truncate long lines
        else:
            box.label(text="No logs yet")

class WHISPERX_PT_output_panel(Panel):
    bl_label = "Transcription Result"
    bl_idname = "WHISPERX_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'WhisperX'
    bl_parent_id = "WHISPERX_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.whisperx_props
        
        if props.transcription_output:
            try:
                # Try to parse JSON and show formatted output
                data = json.loads(props.transcription_output)
                
                # Show basic info
                box = layout.box()
                row = box.row()
                row.label(text="Language:")
                row.label(text=data.get('language', 'Unknown'))
                
                row = box.row()
                row.label(text="Duration:")
                row.label(text=f"{data.get('audio_duration', 0):.1f}s")
                
                row = box.row()
                row.label(text="Processing Time:")
                row.label(text=f"{data.get('processing_time', 0):.1f}s")
                
                # Show transcript
                transcript = data.get('transcript', '')
                if transcript:
                    box = layout.box()
                    box.label(text="Transcript:", icon='TEXT')
                    
                    # Split long text into multiple lines
                    words = transcript.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + word) < 60:
                            current_line += word + " "
                        else:
                            lines.append(current_line.strip())
                            current_line = word + " "
                    if current_line:
                        lines.append(current_line.strip())
                    
                    for line in lines:
                        row = box.row()
                        row.scale_y = 0.9
                        row.label(text=line)
                
                # Show segments info
                segments = data.get('segments', [])
                if segments:
                    box = layout.box()
                    box.label(text=f"Segments: {len(segments)}", icon='SEQUENCE')
                    
                    # Add buttons for output options
                    row = box.row(align=True)
                    row.operator("whisperx.send_to_nla", icon='NLA')
                    row.operator("whisperx.send_to_vse", icon='SEQUENCE')
                    row.operator("whisperx.copy_transcript", icon='COPYDOWN')
                
            except json.JSONDecodeError:
                # Show raw output if not valid JSON
                box = layout.box()
                box.label(text="Raw Output:")
                lines = props.transcription_output.split('\n')[:10]
                for line in lines:
                    if line.strip():
                        row = box.row()
                        row.scale_y = 0.8
                        row.label(text=line[:80])
        else:
            layout.label(text="No transcription results yet")

# Panel for Properties > Tools area
class WHISPERX_PT_tools_panel(Panel):
    bl_label = "WhisperX FLOW"
    bl_idname = "WHISPERX_PT_tools_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.whisperx_props
        
        # Status indicator
        status_box = layout.box()
        row = status_box.row()
        
        # Status icon based on state
        if props.ui_state == STATE_INITIAL:
            row.label(text="", icon='RADIOBUT_OFF')
            status_text = "Service not running"
        elif props.ui_state == STATE_PROCESSING:
            row.label(text="", icon='SORTTIME')
            status_text = props.status_message
        elif props.ui_state == STATE_RUNNING:
            row.label(text="", icon='CHECKMARK')
            status_text = "Service running"
        elif props.ui_state == STATE_MODEL_READY:
            row.label(text="", icon='CHECKMARK')
            status_text = f"Model: {props.loaded_model}"
        elif props.ui_state == STATE_TRANSCRIBED:
            row.label(text="", icon='CHECKMARK')
            status_text = f"Transcription complete"
        else:
            row.label(text="", icon='QUESTION')
            status_text = props.status_message
            
        row.label(text=status_text)
        
        # If transcription is complete, show output options
        if props.ui_state == STATE_TRANSCRIBED:
            # Output options
            box = layout.box()
            box.label(text="Output Options", icon='OUTPUT')
            
            # NLA output
            row = box.row()
            row.label(text="NLA Animation:")
            row = box.row(align=True)
            row.operator("whisperx.send_to_nla", icon='NLA')
            row.operator("whisperx.remove_from_nla", icon='X', text="")
            
            # Video Editor output
            row = box.row()
            row.label(text="Video Editor:")
            row = box.row(align=True)
            row.operator("whisperx.send_to_vse", icon='SEQUENCE')
            row.operator("whisperx.remove_from_vse", icon='X', text="")
            
            # VSE options
            row = box.row()
            row.prop(props, "vse_use_words", text="Use Words (not Segments)")
            
            # Copy transcript button
            row = box.row()
            row.operator("whisperx.copy_transcript", icon='COPYDOWN')
            
            # Try again button
            row = layout.row()
            row.scale_y = 1.2
            row.operator("whisperx.try_again", icon='FILE_REFRESH', text="Try Again")
        else:
            # If not transcribed, show link to main panel
            row = layout.row()
            row.label(text="Use the WhisperX panel in the 3D View sidebar")
            row = layout.row()
            row.label(text="to transcribe audio files.")

classes = [
    WhisperXProperties,
    WHISPERX_OT_locate_script,
    WHISPERX_OT_start_service,
    WHISPERX_OT_stop_service,
    WHISPERX_OT_load_model,
    WHISPERX_OT_transcribe_audio,
    WHISPERX_OT_try_again,
    WHISPERX_OT_clear_logs,
    WHISPERX_OT_toggle_advanced,
    WHISPERX_OT_copy_transcript,
    WHISPERX_OT_send_to_nla,
    WHISPERX_OT_remove_from_nla,
    WHISPERX_OT_send_to_vse,
    WHISPERX_OT_remove_from_vse,
    WHISPERX_OT_add_audio_to_scene,
    WHISPERX_PT_main_panel,
    WHISPERX_PT_logs_panel,
    WHISPERX_PT_output_panel,
    WHISPERX_PT_tools_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.whisperx_props = bpy.props.PointerProperty(type=WhisperXProperties)
    
    # Try to find the script automatically
    script_path = find_whisperx_script()
    if script_path:
        bpy.context.scene.whisperx_props.script_path = script_path

def unregister():
    global ui_update_timer
    
    # Stop UI update timer if running
    if ui_update_timer is not None and bpy.app.timers.is_registered(ui_update_timer_callback):
        bpy.app.timers.unregister(ui_update_timer_callback)
    
    # Clean up any running process
    global whisperx_process
    if whisperx_process and whisperx_process.poll() is None:
        try:
            # Try to send exit command first
            try:
                whisperx_process.stdin.write("exit()\n")
                whisperx_process.stdin.flush()
                time.sleep(0.5)
            except:
                pass
                
            # Then terminate if still running
            if whisperx_process.poll() is None:
                whisperx_process.terminate()
                time.sleep(0.5)
                if whisperx_process.poll() is None:
                    whisperx_process.kill()
        except:
            pass
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    if hasattr(bpy.types.Scene, "whisperx_props"):
        del bpy.types.Scene.whisperx_props

if __name__ == "__main__":
    register()
