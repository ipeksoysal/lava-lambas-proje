
import struct, secrets, os, time
from google.colab import files

MAGIC_SQUARE = [
    [52, 61,  4, 13, 20, 29, 36, 45],
    [14,  3, 62, 51, 46, 35, 30, 19],
    [53, 60,  5, 12, 21, 28, 37, 44],
    [11,  6, 59, 54, 43, 38, 27, 22],
    [55, 58,  7, 10, 23, 26, 39, 42],
    [ 9,  8, 57, 56, 41, 40, 25, 24],
    [50, 63,  2, 15, 18, 31, 34, 47],
    [16,  1, 64, 49, 48, 33, 32, 17],
]
MAGIC_PERM = [0] * 64
for _i in range(8):
    for _j in range(8):
        src = MAGIC_SQUARE[_i][_j] - 1   
        dst = _i * 8 + _j
        MAGIC_PERM[src] = dst

MAGIC_PERM_INV = [0] * 64
for _s, _d in enumerate(MAGIC_PERM):
    MAGIC_PERM_INV[_d] = _s

_T1 = [52,61,4,13,20,29,36,45,14,3,62,51,46,35,30,19,53,60,5,12,21,28,37,
       44,11,6,59,54,43,38,27,22,55,58,7,10,23,26,39,42,9,8,57,56,41,40,25,
       24,50,63,2,15,18,31,34,47,16,1,64,49,48,33,32,17]

_T2 = [45,53,14,21,30,19,44,35,5,13,20,61,36,43,22,27,58,52,11,4,62,54,29,
       39,3,9,23,41,51,46,24,28,59,50,60,6,12,18,47,37,1,10,49,48,34,42,38,
       31,64,55,16,7,32,17,26,33,8,57,56,63,2,15,40,25]

_T2_pos = {v: i for i, v in enumerate(_T2)}
KT_PERM     = [_T2_pos[v] for v in _T1]   # KT_PERM[src] = dst
KT_PERM_INV = [0] * 64
for _s, _d in enumerate(KT_PERM):
    KT_PERM_INV[_d] = _s

KT_PATH = [
     0, 17, 34, 51, 61, 55, 38, 23,  6, 12,
    29, 46, 63, 53, 47, 30, 15,  5, 22, 39,
    45, 62, 52, 37, 54, 44, 27, 21, 31, 14,
     4, 10, 20,  3,  9, 24, 18,  8,  2, 19,
    36, 26, 32, 49, 59, 42, 57, 40, 25, 35,
    41, 56, 50, 60, 43, 58, 48, 33, 16,  1,
    11, 28, 13,  7
]


PACKET_BYTES = 512      
NUM_BLOCKS   = 64         
ROUNDS       = 8          
MOD64        = 2 ** 64
FILE_MAGIC   = b'EMRE\x01' 


def to_blocks(data: bytes) -> list:
    return list(struct.unpack('>64Q', data))

def from_blocks(blocks) -> bytes:
    return struct.pack('>64Q', *[int(b) % MOD64 for b in blocks])

def circ_right(data: bytes, shift: int) -> bytes:
    n = len(data) * 8
    shift %= n
    if shift == 0:
        return data
    val = int.from_bytes(data, 'big')
    val = ((val >> shift) | (val << (n - shift))) & ((1 << n) - 1)
    return val.to_bytes(len(data), 'big')

def circ_left(data: bytes, shift: int) -> bytes:
    n = len(data) * 8
    return circ_right(data, n - shift % n)

def derive_round_keys(master_key: bytes, pkt_idx: int) -> list:
    base_keys = list(struct.unpack('>64Q', master_key[:PACKET_BYTES]))

    round_keys = []
    for r in range(ROUNDS):
        rk = [(k ^ (pkt_idx << 8) ^ r) % MOD64 for k in base_keys]
        round_keys.append(rk)
    return round_keys

def encrypt_round(blocks: list, key: list, shift: int) -> list:
    added = [0] * NUM_BLOCKS
    for src in range(NUM_BLOCKS):
        dst       = MAGIC_PERM[src]
        added[dst] = (blocks[src] + key[src]) % MOD64

    shifted_bytes = circ_right(from_blocks(added), shift)
    shifted       = to_blocks(shifted_bytes)
    
    xored = list(shifted)
    for i in range(63):
        src = KT_PATH[i]
        dst = KT_PATH[i + 1]
        xored[dst] = xored[src] ^ xored[dst]

    return xored


def decrypt_round(blocks: list, key: list, shift: int) -> list:
    unxored = list(blocks)
    for i in range(62, -1, -1):
        src = KT_PATH[i]
        dst = KT_PATH[i + 1]
        unxored[dst] = unxored[src] ^ unxored[dst]

    unshifted_bytes = circ_left(from_blocks(unxored), shift)
    unshifted       = to_blocks(unshifted_bytes)

    unadded = [0] * NUM_BLOCKS
    for dst in range(NUM_BLOCKS):
        src           = MAGIC_PERM_INV[dst]
        unadded[src]  = (unshifted[dst] - key[src]) % MOD64

    return unadded

def encrypt_packet(pkt: bytes, master_key: bytes,
                   pkt_idx: int, shifts: list) -> bytes:
    blocks     = to_blocks(pkt)
    round_keys = derive_round_keys(master_key, pkt_idx)
    for r in range(ROUNDS):
        blocks = encrypt_round(blocks, round_keys[r], shifts[r])
    return from_blocks(blocks)


def decrypt_packet(pkt: bytes, master_key: bytes,
                   pkt_idx: int, shifts: list) -> bytes:
    blocks     = to_blocks(pkt)
    round_keys = derive_round_keys(master_key, pkt_idx)
    for r in range(ROUNDS - 1, -1, -1):
        blocks = decrypt_round(blocks, round_keys[r], shifts[r])
    return from_blocks(blocks)


def encrypt_file(plaintext: bytes, master_key: bytes) -> bytes:
    original_len = len(plaintext)

    pad_len        = (-original_len) % PACKET_BYTES
    padded         = plaintext + bytes(pad_len)
    num_packets    = len(padded) // PACKET_BYTES

    all_shifts = [[secrets.randbelow(4096) for _ in range(ROUNDS)]
                  for _ in range(num_packets)]
    hdr  = FILE_MAGIC
    hdr += struct.pack('>Q', original_len)
    hdr += struct.pack('>I', pad_len)
    hdr += struct.pack('>I', num_packets)
    for shifts in all_shifts:
        for s in shifts:
            hdr += struct.pack('>H', s)

    enc_pkts   = []
    prev_enc   = bytes(PACKET_BYTES)   # initial chain = zero block

    for i in range(num_packets):
        raw_pkt = padded[i * PACKET_BYTES:(i + 1) * PACKET_BYTES]

        if i > 0:
            raw_pkt = bytes(a ^ b for a, b in zip(raw_pkt, prev_enc))

        enc_pkt = encrypt_packet(raw_pkt, master_key, i, all_shifts[i])
        enc_pkts.append(enc_pkt)
        prev_enc = enc_pkt

    return hdr + b''.join(enc_pkts)


def decrypt_file(ciphertext: bytes, master_key: bytes) -> bytes:
    off = 0

    assert ciphertext[off:off+5] == FILE_MAGIC, \
        "❌ Invalid file — not encrypted with Emre's Algorithm."
    off += 5
    original_len = struct.unpack('>Q', ciphertext[off:off+8])[0];  off += 8
    pad_len      = struct.unpack('>I', ciphertext[off:off+4])[0];  off += 4
    num_packets  = struct.unpack('>I', ciphertext[off:off+4])[0];  off += 4

    all_shifts = []
    for _ in range(num_packets):
        shifts = [struct.unpack('>H', ciphertext[off+k*2:off+k*2+2])[0]
                  for k in range(ROUNDS)]
        off += ROUNDS * 2
        all_shifts.append(shifts)

    enc_data = ciphertext[off:]
    dec_pkts = []
    prev_enc = bytes(PACKET_BYTES)

    for i in range(num_packets):
        enc_pkt = enc_data[i * PACKET_BYTES:(i + 1) * PACKET_BYTES]
        dec_pkt = decrypt_packet(enc_pkt, master_key, i, all_shifts[i])

        # Undo reductive chaining
        if i > 0:
            dec_pkt = bytes(a ^ b for a, b in zip(dec_pkt, prev_enc))

        dec_pkts.append(dec_pkt)
        prev_enc = enc_pkt

    return b''.join(dec_pkts)[:original_len]

def main():
    print("📁  Please upload the file you want to encrypt:")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    filename  = list(uploaded.keys())[0]
    plaintext = uploaded[filename]
    print(f"\n✔  Loaded: {filename}  ({len(plaintext):,} bytes)")

    KEY_HEX = (
        "a3f1c2e4b5d6a7f89e0c1d2b3a4f5e6d7c8b9a0f1e2d3c4b5a6f7e8d9c0b1a2f"
        "3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b9a0f1e2d3c4b5a6f7e8d9c0b1a2f3e4d"
        "f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2"
        "b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4"
        "c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3"
        "e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5"
        "d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6"
        "f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8"
        "e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7"
        "a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9"
        "f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8"
        "b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0"
        "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
        "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4"
        "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3"
        "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5"
    )
    master_key = bytes.fromhex(KEY_HEX)
    assert len(master_key) == 512, "Key must be exactly 512 bytes (4096 bits)!"
    print(f"\n🔑  Using shared key (4096 bit): {master_key.hex()[:32]}…")

    print("\n⚙️   Encrypting…")
    t0 = time.perf_counter()
    ciphertext = encrypt_file(plaintext, master_key)
    t1 = time.perf_counter()
    enc_secs   = t1 - t0

    num_pkts   = (len(plaintext) + PACKET_BYTES - 1) // PACKET_BYTES
    print(f"✔  Done in {enc_secs:.2f}s  |  "
          f"{num_pkts} packet(s)  |  "
          f"{len(ciphertext):,} bytes output")

    print("\n⚙️   Decrypting…")
    t0 = time.perf_counter()
    decrypted = decrypt_file(ciphertext, master_key)
    t1 = time.perf_counter()
    dec_secs  = t1 - t0
    print(f"✔  Done in {dec_secs:.2f}s  |  {len(decrypted):,} bytes recovered")

    match = plaintext == decrypted

    print("\n" + "─" * 56)
    print("  INTEGRITY CHECK")
    print("─" * 56)
    print(f"  Original size  : {len(plaintext):,} bytes")
    print(f"  Decrypted size : {len(decrypted):,} bytes")
    print(f"  Files identical: {'✅  YES — Encryption/Decryption verified!' if match else '❌  NO  — Mismatch detected!'}")
    print("─" * 56)
    
    enc_name = filename + ".emre"
    dec_name = "decrypted_" + filename

    with open(enc_name, 'wb') as f:
        f.write(ciphertext)
    with open(dec_name, 'wb') as f:
        f.write(decrypted)

    print(f"\n📥  Downloading encrypted file → {enc_name}")
    files.download(enc_name)

    print(f"📥  Downloading decrypted file → {dec_name}")
    files.download(dec_name)

    print("\n✅  All done!")

main()
