import math

# Dosyayı oku
with open("lava_bits.txt", "r") as f:
    bits = f.read().strip()

n = len(bits)

print("Toplam bit:", n)

# FREKANS TESTİ (0 ve 1 dengesi)

ones = bits.count("1")
zeros = bits.count("0")

print("1 sayısı:", ones)
print("0 sayısı:", zeros)

ratio = ones / n
print("1 oranı:", ratio)

#  ENTROPİ HESABI

p1 = ones / n
p0 = zeros / n

entropy = 0
if p1 > 0:
    entropy -= p1 * math.log2(p1)
if p0 > 0:
    entropy -= p0 * math.log2(p0)

print("Shannon Entropi:", entropy)

# RUNS TESTİ (0-1 geçiş sayısı)

runs = 1
for i in range(1, n):
    if bits[i] != bits[i-1]:
        runs += 1

print("Toplam run sayısı:", runs)