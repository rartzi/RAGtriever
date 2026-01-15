from ragtriever.hashing import blake2b_hex

def test_blake2b_hex_stable():
    a = blake2b_hex(b"hello")
    b = blake2b_hex(b"hello")
    assert a == b
