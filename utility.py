MAIN_CHUNK_SIZE = 2 ** 18

def chunk_seq(N, chunk_size=MAIN_CHUNK_SIZE):
    while N > chunk_size:
        N -= chunk_size
        yield chunk_size
    if N > 0:
        yield N

def grab(generator, n):
    return [next(generator) for _ in range(n)]

def mgrab(generators, n):
    return [
        [next(g) for g in generators]
        for _ in range(n)
    ]
