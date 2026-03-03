import cv2
import hashlib

img = cv2.imread("lava.jpg")

if img is None:
    print("Görsel bulunamadı!")
    exit()

image_bytes = img.tobytes()

current_hash = hashlib.sha3_256(image_bytes).digest()

rounds = 20000  # Arttırılabilir
all_bytes = b""

for _ in range(rounds):
    current_hash = hashlib.sha3_256(current_hash).digest()
    all_bytes += current_hash

bit_string = ''.join(format(byte, '08b') for byte in all_bytes)

with open("lava_bits.txt", "w") as f:
    f.write(bit_string)

print("Bitti.")

print("Toplam bit:", len(bit_string))


