import sys
import torch
import torchaudio
import subprocess

def check_system():
    print("Python Version:", sys.version)
    print("\nPyTorch:")
    print("Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
    
    print("\nTorchaudio:")
    print("Version:", torchaudio.__version__)
    
    print("\nFFmpeg Check:")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        print(result.stdout.split('\n')[0])
    except Exception as e:
        print("FFmpeg not found:", e)

if __name__ == "__main__":
    check_system()