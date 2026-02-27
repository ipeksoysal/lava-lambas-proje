import cv2
import hashlib

#  Görüntüyü oku
img = cv2.imread("lava.jpg")

if img is None:
    print("Görsel bulunamadı!")
    exit()

# Görüntüyü byte'a çevir
image_bytes = img.tobytes()

#  İlk SHA3 hash
current_hash = hashlib.sha3_256(image_bytes).digest()

# Hash zinciri
rounds = 20000  # Artırılabilir
all_bytes = b""

for _ in range(rounds):
    current_hash = hashlib.sha3_256(current_hash).digest()
    all_bytes += current_hash

#  Bit stringe çevir (tek satır)
bit_string = ''.join(format(byte, '08b') for byte in all_bytes)

# Dosyaya yaz
with open("lava_bits.txt", "w") as f:
    f.write(bit_string)

print("Bitti.")

print("Toplam bit:", len(bit_string))
