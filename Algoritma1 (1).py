# poly_cipher_fix.py
from decimal import Decimal, getcontext, ROUND_HALF_UP
import base64

getcontext().prec = 60
MOD = 256

E_COEFFS = [
    Decimal("0.03"),
    Decimal("-0.59"),
    Decimal("4.63"),
    Decimal("-19.75"),
    Decimal("50.12"),
    Decimal("-77.17"),
    Decimal("70.43"),
    Decimal("0"),
    Decimal("0"),
    Decimal("0"),
]

def poly_eval_decimal(coeffs, x_int: int) -> Decimal:
    x = Decimal(x_int)
    y = Decimal(0)
    for a in coeffs:      # Horner
        y = y * x + a
    return y

def round_to_int(d: Decimal) -> int:
    return int(d.to_integral_value(rounding=ROUND_HALF_UP))

def tag_from_x(x: int) -> int:
    y = poly_eval_decimal(E_COEFFS, x)
    return round_to_int(y) % MOD

def encrypt_text_to_b64(plaintext: str) -> str:
    data = plaintext.encode("utf-8")
    out = bytearray()
    for x in data:
        t = tag_from_x(x)
        c = x ^ t
        # 2 byte saklıyoruz: (tag, c)
        out.append(t)
        out.append(c)
    return base64.b64encode(bytes(out)).decode("ascii")

def decrypt_b64_to_text(cipher_b64: str) -> str:
    raw = base64.b64decode(cipher_b64.encode("ascii"))
    if len(raw) % 2 != 0:
        raise ValueError("Cipher uzunluğu bozuk (çift olmalı).")
    out = bytearray()
    for i in range(0, len(raw), 2):
        t = raw[i]
        c = raw[i+1]
        x = c ^ t
        # İsteğe bağlı doğrulama:
        if tag_from_x(x) != t:
            # Bozulma/yanlış ana veri tespiti
            raise ValueError(f"Doğrulama hatası: index {i//2}")
        out.append(x)
    return out.decode("utf-8", errors="strict")

def main():
    while True:
        mode = input("\n[E]ncrypt / [D]ecrypt / [Q]uit: ").strip().lower()
        if mode == "q":
            break
        if mode == "e":
            pt = input("Metin gir: ")
            c = encrypt_text_to_b64(pt)
            print("\nŞifreli (Base64):")
            print(c)
        elif mode == "d":
            c = input("Şifreli Base64 gir: ").strip()
            pt = decrypt_b64_to_text(c)
            print("\nÇözülmüş metin:")
            print(pt)
        else:
            print("Geçersiz seçim.")

if __name__ == "__main__":
    main()
