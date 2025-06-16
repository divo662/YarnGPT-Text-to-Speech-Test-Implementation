import secrets
import os

def generate_secret_key(file_path='.secret_key'):
    """
    Generate a secure random secret key
    
    Args:
        file_path (str): Path to save the secret key
    
    Returns:
        str: Generated secret key
    """
    # Generate a 32-byte (256-bit) hex-encoded secret key
    secret_key = secrets.token_hex(32)
    
    # Save the key to a file
    try:
        with open(file_path, 'w') as f:
            f.write(secret_key)
        print(f"Secret key saved to {file_path}")
    except Exception as e:
        print(f"Error saving secret key: {e}")
    
    return secret_key

def load_secret_key(file_path='.secret_key'):
    """
    Load secret key from file, generate if not exists
    
    Args:
        file_path (str): Path to secret key file
    
    Returns:
        str: Secret key
    """
    try:
        # Try to read existing secret key
        with open(file_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Generate and save new secret key if file doesn't exist
        return generate_secret_key(file_path)

# If you run this script directly, it will generate and print a new secret key
if __name__ == '__main__':
    key = generate_secret_key()
    print("Generated Secret Key:", key)