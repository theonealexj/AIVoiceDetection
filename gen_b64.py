import os
import base64

def generate_sample():
    # Target the specific file the user mentioned
    target_file = r"c:\Users\Alex\Downloads\sample voice 1.mp3"
    
    if not os.path.exists(target_file):
        print(f"Error: File not found: {target_file}")
        return

    print(f"Encoding file: {target_file}")
    
    with open(target_file, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode('utf-8')
        
    with open("sample_base64.txt", "w") as out:
        out.write(b64)
        
    print(f"Successfully wrote base64 to sample_base64.txt (Length: {len(b64)})")

if __name__ == "__main__":
    generate_sample()
