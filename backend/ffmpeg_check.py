import subprocess
import sys

def check_ffmpeg():
    try:
        # Try to run ffmpeg and capture its version
        result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        print("FFmpeg version:")
        print(result.stdout.splitlines()[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg check failed: {e}")
        return False

if __name__ == "__main__":
    if check_ffmpeg():
        print("FFmpeg is installed and working correctly.")
    else:
        print("FFmpeg is not properly installed or configured.")
        sys.exit(1)