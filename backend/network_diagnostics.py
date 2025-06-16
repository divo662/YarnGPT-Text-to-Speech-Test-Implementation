# network_diagnostics.py
import requests
import socket

def check_internet_connection():
    try:
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def get_network_info():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"Hostname: {hostname}")
    print(f"Local IP: {local_ip}")
    print(f"Internet Connection: {'Available' if check_internet_connection() else 'Unavailable'}")

if __name__ == "__main__":
    get_network_info()