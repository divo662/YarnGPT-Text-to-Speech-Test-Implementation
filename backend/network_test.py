# network_test.py
import socket

def check_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Connection error: {e}")
        return False

# Test local connections
print("Localhost test:")
print("5000 port open:", check_port('127.0.0.1', 5000))
print("5000 port open:", check_port('localhost', 5000))

# Get local IP
import socket
local_ip = socket.gethostbyname(socket.gethostname())
print(f"\nLocal IP: {local_ip}")
print(f"{local_ip} port open:", check_port(local_ip, 5000))