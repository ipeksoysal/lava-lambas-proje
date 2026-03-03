import base64
import math
import numpy as np
import secrets
from typing import List, Tuple

P1, P2, P3 = 269, 521, 839
P_FINAL = 1629

ROUND_MODS  = {1: P1, 2: P2, 3: P3}
ROUND_SEEDS = {1: P2, 2: P3, 3: P1}  # tur1->p2, tur2->p3, tur3->p1
MIX_ITERS = 1

DEFAULT_TEXT = (
    'Sometimes it is the people no one imagines anything of who do the things that no one can imagine.'
    'We can only see a short distance ahead, but we can see plenty there that needs to be done.'
    "I propose to consider the question, 'Can machines think?'"
)

def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    if pad_len == 0:
        pad_len = block_size
    return data + bytes([pad_len]) * pad_len

def pkcs7_unpad(data: bytes, block_size: int = 16) -> bytes:
    if len(data) == 0 or len(data) % block_size != 0:
        raise ValueError("Padding hatası: uzunluk geçersiz.")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size:
        raise ValueError("Padding hatası: pad_len geçersiz.")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("Padding hatası: padding baytları tutmuyor.")
    return data[:-pad_len]

def chunk_n(data: bytes, n: int) -> List[bytes]:
    assert len(data) % n == 0
    return [data[i:i+n] for i in range(0, len(data), n)]

def bytes16_to_matrix(block16: bytes) -> np.ndarray:
    arr = np.frombuffer(block16, dtype=np.uint8).astype(int)
    return arr.reshape(4, 4)

def matrix_to_words16(S: np.ndarray) -> List[int]:
    return [int(x) for x in S.reshape(-1).tolist()]

def words16_to_matrix(words: List[int]) -> np.ndarray:
    return np.array([int(x) for x in words], dtype=int).reshape(4, 4)

def serialize_words16_le_u16(words: List[int]) -> bytes:
    out = bytearray()
    for w in words:
        w = int(w)
        if not (0 <= w <= 65535):
            raise ValueError("serialize_words: aralık dışı.")
        out += w.to_bytes(2, "little")
    return bytes(out)  # 32 byte

def parse_words16_le_u16(buf32: bytes) -> List[int]:
    if len(buf32) != 32:
        raise ValueError("parse_words: blok 32 byte olmalı.")
    words = []
    for i in range(0, 32, 2):
        words.append(int.from_bytes(buf32[i:i+2], "little"))
    return words

def _u64(x: int) -> int:
    return x & 0xFFFFFFFFFFFFFFFF

class XorShift128Plus:
    def __init__(self, seed_bytes: bytes):
        if len(seed_bytes) < 16:
            seed_bytes = seed_bytes + b"\x00" * (16 - len(seed_bytes))
        s0 = int.from_bytes(seed_bytes[:8], "little")
        s1 = int.from_bytes(seed_bytes[8:16], "little")
        if s0 == 0 and s1 == 0:
            s1 = 1
        self.s0 = _u64(s0)
        self.s1 = _u64(s1)

    def next_u64(self) -> int:
        x = self.s0
        y = self.s1
        self.s0 = y
        x ^= _u64(x << 23)
        x ^= _u64(x >> 17)
        x ^= y
        x ^= _u64(y >> 26)
        self.s1 = x
        return _u64(self.s0 + self.s1)

    def stream_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out += self.next_u64().to_bytes(8, "little")
        return bytes(out[:n])

def derive_block_keys(master_key: bytes, block_index: int) -> Tuple[List[int], List[int]]:
    gen = XorShift128Plus(master_key)
    skip = block_index * 32
    _ = gen.stream_bytes(skip)
    k = gen.stream_bytes(32)
    k1 = list(k[:16])
    k2 = list(k[16:])
    return k1, k2

def whitening_enc(S: np.ndarray, k11_16: List[int], p: int) -> np.ndarray:
    K = np.array(k11_16, dtype=int).reshape(4, 4) % p
    return (S + K) % p

def whitening_dec(S: np.ndarray, k11_16: List[int], p: int) -> np.ndarray:
    K = np.array(k11_16, dtype=int).reshape(4, 4) % p
    return (S - K) % p

PERM_IDX = [
    3,  13, 14, 0,
    8,  6,  5,  11,
    4,  10, 9,  7,
    15, 1,  2,  12
]
INV_PERM_IDX = [0]*16
for out_pos, in_pos in enumerate(PERM_IDX):
    INV_PERM_IDX[in_pos] = out_pos

def permute(S: np.ndarray) -> np.ndarray:
    s = S.reshape(-1)
    return s[PERM_IDX].reshape(4, 4)

def inv_permute(S: np.ndarray) -> np.ndarray:
    s = S.reshape(-1)
    return s[INV_PERM_IDX].reshape(4, 4)

def collatz_bits_once(seed: int) -> List[int]:
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

def expand_collatz_bits(seed: int, length: int) -> np.ndarray:
    out = []
    s = int(seed)
    while len(out) < length:
        out.extend(collatz_bits_once(s))
        s = (3 * s + 1) % (2**31) + 2
        if len(out) == 0:
            out.extend([0, 1, 0, 1, 1, 0, 1, 0])
    return np.array(out[:length], dtype=int)

def collatz_mask_elements(seed: int, p: int, count: int = 16) -> np.ndarray:
    bits_per = math.ceil(math.log2(p))
    bits = expand_collatz_bits(seed, count * bits_per)
    elems = []
    for i in range(count):
        chunk = bits[i*bits_per:(i+1)*bits_per]
        val = 0
        for b in chunk:
            val = (val << 1) | int(b)
        elems.append(val % p)
    return np.array(elems, dtype=int)

def shift_vec(v: np.ndarray) -> np.ndarray:
    return np.roll(v, -1)

def F_of_v(v: np.ndarray, k_v: np.ndarray, c_bits: np.ndarray, p: int) -> np.ndarray:
    return (v * shift_vec(v) + k_v + c_bits) % p

def mix_uv_enc(S: np.ndarray, k11_16: List[int], p: int, seed: int, iters: int) -> np.ndarray:
    s = S.reshape(-1).astype(int)
    u = s[:8].copy()
    v = s[8:].copy()
    kvec = np.array(k11_16, dtype=int) % p
    k_v = kvec[8:]

    for t in range(iters):
        c = expand_collatz_bits(seed + t, 8)
        u_next = v
        v_next = (u + F_of_v(v, k_v, c, p)) % p
        u, v = u_next, v_next

    return np.concatenate([u, v]).reshape(4, 4)

def mix_uv_dec(S: np.ndarray, k11_16: List[int], p: int, seed: int, iters: int) -> np.ndarray:
    s = S.reshape(-1).astype(int)
    u = s[:8].copy()
    v = s[8:].copy()
    kvec = np.array(k11_16, dtype=int) % p
    k_v = kvec[8:]

    for t in reversed(range(iters)):
        u_next, v_next = u, v
        v_prev = u_next
        c = expand_collatz_bits(seed + t, 8)
        u_prev = (v_next - F_of_v(v_prev, k_v, c, p)) % p
        u, v = u_prev, v_prev

    return np.concatenate([u, v]).reshape(4, 4)

def inv_mod(a, p: int) -> int:
    aa = int(a) % int(p)
    if aa == 0:
        raise ZeroDivisionError("0'ın modüler tersi yok.")
    return pow(aa, int(p) - 2, int(p))  # p asal

def mat_inv_mod(A: np.ndarray, p: int) -> np.ndarray:
    A = (A % p).astype(int)
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    aug = np.concatenate([A, I], axis=1).astype(int)

    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, n):
            if int(aug[i, c]) % p != 0:
                pivot = i
                break
        if pivot is None:
            raise ValueError("Matris terslenemez (pivot yok).")
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]

        piv = int(aug[r, c]) % p
        inv_piv = inv_mod(piv, p)
        aug[r] = (aug[r] * inv_piv) % p

        for i in range(n):
            if i != r:
                factor = int(aug[i, c]) % p
                if factor != 0:
                    aug[i] = (aug[i] - factor * aug[r]) % p
        r += 1

    return aug[:, n:] % p

def is_invertible_mod_p(A: np.ndarray, p: int) -> bool:
    try:
        _ = mat_inv_mod(A, p)
        return True
    except Exception:
        return False

def derive_invertible_A(k2_16: List[int], p: int) -> np.ndarray:
    K = np.array(k2_16, dtype=int).reshape(4, 4) % p
    I = np.eye(4, dtype=int)
    for t in range(0, min(p, 4096)):
        A = (I + K + t * I) % p
        if is_invertible_mod_p(A, p):
            return A
    raise ValueError("Invertible A üretilemedi.")

def diffusion_enc(S: np.ndarray, k2_16: List[int], p: int) -> np.ndarray:
    A = derive_invertible_A(k2_16, p)
    out = np.zeros_like(S, dtype=int)
    for j in range(4):
        col = S[:, j].reshape(4, 1).astype(int)
        out[:, j] = ((A @ col) % p).flatten()
    return out

def diffusion_dec(S: np.ndarray, k2_16: List[int], p: int) -> np.ndarray:
    A = derive_invertible_A(k2_16, p)
    Ainv = mat_inv_mod(A, p)
    out = np.zeros_like(S, dtype=int)
    for j in range(4):
        col = S[:, j].reshape(4, 1).astype(int)
        out[:, j] = ((Ainv @ col) % p).flatten()
    return out

def round5_enc(S: np.ndarray, p: int, seed: int) -> np.ndarray:
    M = collatz_mask_elements(seed, p, count=16).reshape(4, 4)
    return (S + M) % p

def round5_dec(S: np.ndarray, p: int, seed: int) -> np.ndarray:
    M = collatz_mask_elements(seed, p, count=16).reshape(4, 4)
    return (S - M) % p

def apply_round_enc(S: np.ndarray, r: int, k11: List[int], k12: List[int]) -> np.ndarray:
    p = ROUND_MODS[r]
    seed = ROUND_SEEDS[r]
    S = whitening_enc(S, k11, p)
    S = permute(S)
    S = mix_uv_enc(S, k11, p, seed, MIX_ITERS)
    S = diffusion_enc(S, k12, p)
    S = round5_enc(S, p, seed)
    return S

def apply_round_dec(S: np.ndarray, r: int, k11: List[int], k12: List[int]) -> np.ndarray:
    p = ROUND_MODS[r]
    seed = ROUND_SEEDS[r]
    S = round5_dec(S, p, seed)
    S = diffusion_dec(S, k12, p)
    S = mix_uv_dec(S, k11, p, seed, MIX_ITERS)
    S = inv_permute(S)
    S = whitening_dec(S, k11, p)
    return S

def bytes_to_bits(byte_data: bytes) -> np.ndarray:
    bits = []
    for b in byte_data:
        for k in range(7, -1, -1):
            bits.append((b >> k) & 1)
    return np.array(bits, dtype=int)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = [int(x) for x in bits]
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for bit in bits[i:i+8]:
            val = (val << 1) | bit
        out.append(val & 0xFF)
    return bytes(out)

def final_xor(data: bytes, seed: int = P_FINAL) -> bytes:
    b = bytes_to_bits(data)
    m = expand_collatz_bits(seed, len(b))
    return bits_to_bytes(b ^ m)

def encrypt_block(block16: bytes, master_key: bytes, block_index: int) -> bytes:
    k11, k12 = derive_block_keys(master_key, block_index)
    S = bytes16_to_matrix(block16)

    for r in (1, 2, 3):
        S = apply_round_enc(S, r, k11, k12)

    # S artık F_{p3} içinde. 16 word -> u16 serialize (32 byte)
    words = matrix_to_words16(S)
    return serialize_words16_le_u16(words)

def decrypt_block(block32: bytes, master_key: bytes, block_index: int) -> bytes:
    k11, k12 = derive_block_keys(master_key, block_index)
    words = parse_words16_le_u16(block32)
    S = words16_to_matrix(words) % P3

    for r in (3, 2, 1):
        S = apply_round_dec(S, r, k11, k12)

    pt16 = bytes([(int(x) % 256) for x in S.reshape(-1).tolist()])
    return pt16

def encrypt_message(plaintext: str, master_key: bytes) -> bytes:
    pt = pkcs7_pad(plaintext.encode("utf-8"), 16)
    blocks16 = chunk_n(pt, 16)

    ct_blocks32 = []
    for i, b16 in enumerate(blocks16):
        ct_blocks32.append(encrypt_block(b16, master_key, i))

    ct = b"".join(ct_blocks32)       
    ct_final = final_xor(ct, P_FINAL) 
    return ct_final

def decrypt_message(cipher_final: bytes, master_key: bytes) -> str:
    ct = final_xor(cipher_final, P_FINAL)
    if len(ct) % 32 != 0:
        raise ValueError("Ciphertext uzunluğu 32'nin katı değil (format bozuk).")

    blocks32 = chunk_n(ct, 32)
    pt_blocks16 = []
    for i, b32 in enumerate(blocks32):
        pt_blocks16.append(decrypt_block(b32, master_key, i))

    pt_padded = b"".join(pt_blocks16)
    pt = pkcs7_unpad(pt_padded, 16)
    return pt.decode("utf-8", errors="strict")

if __name__ == "__main__":
    print("\n=== BLOK ŞİFRE (3 tur) + Final Collatz ===")
    print("1) Şifrele")
    print("2) Şifrele + Aynı anahtarla otomatik çöz (DEMO)")
    print("3) Çöz (Base64 ciphertext + HEX key)")
    choice = input("Seçim: ").strip()

    if choice not in ("1", "2", "3"):
        raise SystemExit("Geçersiz seçim.")

    if choice in ("1", "2"):
        msg = input("\nMetni gir (boş bırak = varsayılan alıntı): ").strip()
        if msg == "":
            msg = DEFAULT_TEXT

        master_key = secrets.token_bytes(32)
        key_hex = master_key.hex()

        ct_final = encrypt_message(msg, master_key)
        ct_b64 = base64.b64encode(ct_final).decode("ascii")
        ct_hex = ct_final.hex()

        print("\n[MASTER KEY HEX] (sakla):")
        print(key_hex)

        print("\n[CIPHERTEXT Base64] (final sonrası):")
        print(ct_b64)

        print("\n[CIPHERTEXT HEX] (final sonrası):")
        print(ct_hex)

        if choice == "2":
            pt = decrypt_message(base64.b64decode(ct_b64), master_key)
            print("\n[DEMO - ÇÖZÜLEN METİN]:")
            print(pt)

    else:
        b64 = input("\nCiphertext Base64 gir: ").strip()
        key_hex = input("Master key HEX gir: ").strip()

        master_key = bytes.fromhex(key_hex)
        cipher_final = base64.b64decode(b64)

        pt = decrypt_message(cipher_final, master_key)
        print("\n[ÇÖZÜLEN METİN]:")

        print(pt)
