import numpy as np

P1, P2, P3 = 269, 521, 839
P_FINAL = 1629  

ROUND_SEEDS = {1: P2, 2: P3, 3: P1}

K11_BYTES = [
    71, 101, 114, 195,
    167, 101, 107, 116,
    101, 110,  32, 101,
    118, 114, 101, 110
]

K12_BYTES = [
    177, 108,  97, 114,
     52,  89, 201,  33,
    145,  66,  72, 250,
     19,  88, 142,  61
]

MIX_ITERS = 1  

def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    if pad_len == 0:
        pad_len = block_size
    return data + bytes([pad_len]) * pad_len

def block_to_state(block16: bytes) -> np.ndarray:
    arr = np.frombuffer(block16, dtype=np.uint8).astype(int)
    return arr.reshape(4, 4)

def kbytes_to_matrix(k16: list[int]) -> np.ndarray:
    return np.array(k16, dtype=int).reshape(4, 4)

def whitening(S: np.ndarray, K: np.ndarray, p: int) -> np.ndarray:
    return (S + K) % p

def magic_square_permute(S: np.ndarray) -> np.ndarray:
    s = S.reshape(-1)
    idx = [
        3,  13, 14, 0,
        8,  6,  5,  11,
        4,  10, 9,  7,
        15, 1,  2,  12
    ]
    return s[idx].reshape(4, 4)

def collatz_bits_once(seed: int) -> list[int]:
    x = int(seed)
    bits = []
    while x != 1:
        if x % 2 == 0:
            bits.append(1)
            x //= 2
        else:
            bits.append(0)
            x = 3 * x + 1
    return bits[:-4] if len(bits) > 4 else []

def expand_collatz_mask(seed: int, length: int) -> np.ndarray:
    out = []
    s = int(seed)
    while len(out) < length:
        out.extend(collatz_bits_once(s))
        s = (3 * s + 1) % (2**31) + 2
        if len(out) == 0:
            out.extend([0, 1, 0, 1, 1, 0, 1, 0]) 
    return np.array(out[:length], dtype=int)

def shift_vec(v: np.ndarray) -> np.ndarray:
    return np.roll(v, -1)

def F_of_v(v: np.ndarray, k_v: np.ndarray, c: np.ndarray, p: int) -> np.ndarray:
    return (v * shift_vec(v) + k_v + c) % p

def dual_state_mix(S: np.ndarray, k11_16: list[int], p: int, round_seed: int, iters: int = 1) -> np.ndarray:
    s = S.reshape(-1).astype(int)
    u = s[:8].copy()
    v = s[8:].copy()

    kvec = np.array(k11_16, dtype=int)
    k_v = kvec[8:]  

    for t in range(iters):
        c = expand_collatz_mask(round_seed + t, 8)
        u_next = v
        v_next = (u + F_of_v(v, k_v, c, p)) % p
        u, v = u_next, v_next

    return np.concatenate([u, v]).reshape(4, 4)

def inv_mod(a, p: int) -> int:
    aa = int(a) % int(p)
    if aa == 0:
        raise ZeroDivisionError("0'ın modüler tersi yoktur.")
    return pow(aa, int(p) - 2, int(p))

def is_invertible_mod_p(A: np.ndarray, p: int) -> bool:
    A = (A % p).astype(int).copy()
    n = A.shape[0]
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, n):
            if int(A[i, c]) % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]

        piv = int(A[r, c]) % p
        inv_piv = inv_mod(piv, p)
        A[r] = (A[r] * inv_piv) % p

        for i in range(n):
            if i != r:
                factor = int(A[i, c]) % p
                if factor != 0:
                    A[i] = (A[i] - factor * A[r]) % p
        r += 1
    return r == n

def derive_invertible_A(k2_16: list[int], p: int) -> np.ndarray:
    K = np.array(k2_16, dtype=int).reshape(4, 4)
    I = np.eye(4, dtype=int)
    for t in range(0, min(p, 2048)):
        A = (I + K + t * I) % p
        if is_invertible_mod_p(A, p):
            return A
    raise ValueError("Invertible A üretilemedi.")

def diffusion_columnwise(S: np.ndarray, k2_16: list[int], p: int) -> tuple[np.ndarray, np.ndarray]:
    A = derive_invertible_A(k2_16, p)
    S_out = np.zeros_like(S, dtype=int)
    for j in range(4):
        col = S[:, j].reshape(4, 1).astype(int)
        new_col = (A @ col) % p
        S_out[:, j] = new_col.flatten()
    return S_out, A
    
def bytes_to_bits(byte_list: list[int]) -> np.ndarray:
    bits = []
    for b in byte_list:
        b = int(b) & 0xFF
        for k in range(7, -1, -1):
            bits.append((b >> k) & 1)
    return np.array(bits, dtype=int)

def bits_to_bytes(bits: np.ndarray) -> list[int]:
    bits = [int(x) for x in bits]
    out = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        val = 0
        for bit in chunk:
            val = (val << 1) | bit
        out.append(val)
    return out

def round_bit_mix(S: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_bytes = (S % 256).astype(int).reshape(-1).tolist() 
    state_bits = bytes_to_bits(state_bytes)                 
    mask_bits = expand_collatz_mask(seed, len(state_bits))   
    mixed_bits = state_bits ^ mask_bits
    return mixed_bits, state_bits, mask_bits
    
if __name__ == "__main__":
    while True:
        msg = input("Mesajı gir (çık için 'q'): ").strip()
        if msg.lower() == "q":
            raise SystemExit
        if msg == "":
            print("Boş mesaj olmaz. Lütfen bir metin gir.")
            continue
        break

    raw = msg.encode("utf-8")
    data = pkcs7_pad(raw, 16)

    block1 = data[:16]
    S1 = block_to_state(block1)

    print("\n" + "=" * 60)
    print("BLOK 1 ve DURUM MATRİSİ (S1)")
    print("=" * 60)
    print("Blok 1 (16 byte):", list(block1))
    print("S1:")
    print(S1)

    print("\n" + "=" * 60)
    print(f"TUR 1 (mod {P1})")
    print("=" * 60)

    K11 = kbytes_to_matrix(K11_BYTES)
    S1_w = whitening(S1, K11, P1)
    print("\n[Katman 1] Anahtar Beyazlatma:")
    print("K11:")
    print(K11)
    print("S1_w:")
    print(S1_w)

    S1_p = magic_square_permute(S1_w)
    print("\n[Katman 2] Sihirli Kare Permütasyonu:")
    print("S1_p:")
    print(S1_p)

    S1_m = dual_state_mix(S1_p, K11_BYTES, P1, ROUND_SEEDS[1], iters=MIX_ITERS)
    print(f"\n[Katman 3] u,v Nonlineer Karıştırma (iter={MIX_ITERS}):")
    print("S1_m:")
    print(S1_m)

    S1_d, A = diffusion_columnwise(S1_m, K12_BYTES, P1)
    print("\n[Katman 4] Matris Tabanlı Difüzyon:")
    print("A (invertible):")
    print(A)
    print("S1_d:")
    print(S1_d)


    mixed_bits, state_bits, mask_bits = round_bit_mix(S1_d, seed=ROUND_SEEDS[1])
    mixed_bytes = bits_to_bytes(mixed_bits)

    print("\n" + "=" * 60)
    print("[Katman 5] Tur Sonu Bit Karıştırma (Collatz XOR)")
    print("=" * 60)
    print(f"Collatz seed (Tur1 için): {ROUND_SEEDS[1]}")
    print("State bytes (S1_d mod 256):", (S1_d % 256).astype(int).reshape(-1).tolist())
    print("State bits (ilk 64):", state_bits[:64].tolist())
    print("Mask  bits (ilk 64):", mask_bits[:64].tolist())
    print("XOR   bits (ilk 64):", mixed_bits[:64].tolist())

    print("XOR sonucu bytes (16 byte):", mixed_bytes)
