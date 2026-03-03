import numpy as np

print("RUN VERSION: 2026-02-26 / round5-debug-v1")  

P1, P2, P3 = 269, 521, 839
ROUND_MODS = {1: P1, 2: P2, 3: P3}
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

def bytes16_list_to_state(b16: list[int]) -> np.ndarray:
    return np.array([int(x) & 0xFF for x in b16], dtype=int).reshape(4, 4)

def bytes16_to_state(block16: bytes) -> np.ndarray:
    arr = np.frombuffer(block16, dtype=np.uint8).astype(int)
    return arr.reshape(4, 4)

def whitening(S: np.ndarray, k11_16: list[int], p: int) -> np.ndarray:
    K = np.array(k11_16, dtype=int).reshape(4, 4)
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
        raise ZeroDivisionError
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

def diffusion_columnwise(S: np.ndarray, k2_16: list[int], p: int) -> np.ndarray:
    A = derive_invertible_A(k2_16, p)
    S_out = np.zeros_like(S, dtype=int)
    for j in range(4):
        col = S[:, j].reshape(4, 1).astype(int)
        S_out[:, j] = ((A @ col) % p).flatten()
    return S_out

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
        val = 0
        for bit in bits[i:i+8]:
            val = (val << 1) | bit
        out.append(val)
    return out

def round5_mix(S: np.ndarray, seed: int):
    state_bytes = (S % 256).astype(int).reshape(-1).tolist()    
    state_bits  = bytes_to_bits(state_bytes)                   
    mask_bits   = expand_collatz_mask(seed, len(state_bits))    
    xor_bits    = state_bits ^ mask_bits
    out_bytes   = bits_to_bytes(xor_bits)                       
    return state_bytes, state_bits, mask_bits, xor_bits, out_bytes

def apply_round(state_bytes16: list[int], r: int):
    p = ROUND_MODS[r]
    seed = ROUND_SEEDS[r]

    S = bytes16_list_to_state(state_bytes16)
    S = whitening(S, K11_BYTES, p)
    S = magic_square_permute(S)
    S = dual_state_mix(S, K11_BYTES, p, seed, iters=MIX_ITERS)
    S = diffusion_columnwise(S, K12_BYTES, p)

    return round5_mix(S, seed)

if __name__ == "__main__":
    msg = input("Mesajı gir: ").strip()
    if msg == "":
        raise SystemExit("Boş mesaj girildi.")

    data = pkcs7_pad(msg.encode("utf-8"), 16)
    block1 = data[:16]
    state = list(block1)

    print("\n" + "=" * 60)
    print("BLOK 1 (girdi) 16 byte:", state)
    print("=" * 60)

    for r in (1, 2, 3):
        state_bytes, state_bits, mask_bits, xor_bits, out_bytes = apply_round(state, r)

        print("\n" + "=" * 60)
        print(f"TUR {r} (mod {ROUND_MODS[r]}) - KATMAN 5 DETAY")
        print("=" * 60)
        print("Collatz seed:", ROUND_SEEDS[r])
        print("State bytes (difüzyon sonrası S mod 256):", state_bytes)
        print("Mask bits  (ilk 64):", mask_bits[:64].tolist())
        print("XOR  bits  (ilk 64):", xor_bits[:64].tolist())
        print("Tur çıkışı (16 byte):", out_bytes)

        state = out_bytes 

    print("\n" + "=" * 60)
    print("3. TUR SONU ÇIKTI (16 byte)")
    print("=" * 60)
    print(state)

    print("\n" + "=" * 60)
    print("3. TUR SONU DURUM MATRİSİ (4x4)")
    print("=" * 60)

    print(bytes16_list_to_state(state))

