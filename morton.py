from typing import Tuple

class Morton2D:
    """
    Stateless 2D bit-interleaving (64-bit).
    Maps 32-bit (x, y) <-> 64-bit code.
    """
    # 64-bit Masks
    MASK_1 = 0x00000000FFFFFFFF
    MASK_2 = 0x0000FFFF0000FFFF
    MASK_3 = 0x00FF00FF00FF00FF
    MASK_4 = 0x0F0F0F0F0F0F0F0F
    MASK_5 = 0x3333333333333333
    MASK_6 = 0x5555555555555555

    @staticmethod
    def spread(n: int) -> int:
        n &= Morton2D.MASK_1
        n = (n | (n << 16)) & Morton2D.MASK_2
        n = (n | (n << 8))  & Morton2D.MASK_3
        n = (n | (n << 4))  & Morton2D.MASK_4
        n = (n | (n << 2))  & Morton2D.MASK_5
        n = (n | (n << 1))  & Morton2D.MASK_6
        return n

    @staticmethod
    def compact(n: int) -> int:
        n &= Morton2D.MASK_6
        n = (n | (n >> 1)) & Morton2D.MASK_5
        n = (n | (n >> 2)) & Morton2D.MASK_4
        n = (n | (n >> 4)) & Morton2D.MASK_3
        n = (n | (n >> 8)) & Morton2D.MASK_2
        n = (n | (n >> 16)) & Morton2D.MASK_1
        return n

    @staticmethod
    def encode(coords: Tuple[int, ...]) -> int:
        return (Morton2D.spread(coords[1]) << 1) | Morton2D.spread(coords[0])

    @staticmethod
    def decode(code: int) -> Tuple[int, int]:
        return (Morton2D.compact(code), Morton2D.compact(code >> 1))

    @staticmethod
    def get_neighbor(code: int, level: int, dx: int, dy: int, max_level: int = 29) -> int:
        x, y = Morton2D.decode(code)
        size = 1 << (max_level - level)
        nx, ny = x + dx * size, y + dy * size
        limit = 1 << max_level
        if nx < 0 or nx >= limit or ny < 0 or ny >= limit:
            return -1
        return Morton2D.encode((nx, ny))


class Morton3D:
    """
    Stateless 3D bit-interleaving (64-bit).
    Maps 21-bit (x, y, z) <-> 63-bit code.
    """
    MASK_1 = 0x1FFFFF 
    MASK_2 = 0x1F00000000FFFF
    MASK_3 = 0x1F0000FF0000FF
    MASK_4 = 0x100F00F00F00F00F
    MASK_5 = 0x10C30C30C30C30C3
    MASK_6 = 0x1249249249249249

    @staticmethod
    def spread(n: int) -> int:
        n &= Morton3D.MASK_1
        n = (n | (n << 32)) & Morton3D.MASK_2
        n = (n | (n << 16)) & Morton3D.MASK_3
        n = (n | (n << 8))  & Morton3D.MASK_4
        n = (n | (n << 4))  & Morton3D.MASK_5
        n = (n | (n << 2))  & Morton3D.MASK_6
        return n

    @staticmethod
    def compact(n: int) -> int:
        n &= Morton3D.MASK_6
        n = (n | (n >> 2))  & Morton3D.MASK_5
        n = (n | (n >> 4))  & Morton3D.MASK_4
        n = (n | (n >> 8))  & Morton3D.MASK_3
        n = (n | (n >> 16)) & Morton3D.MASK_2
        n = (n | (n >> 32)) & Morton3D.MASK_1
        return n

    @staticmethod
    def encode(coords: Tuple[int, ...]) -> int:
        return (Morton3D.spread(coords[2]) << 2) | (Morton3D.spread(coords[1]) << 1) | Morton3D.spread(coords[0])

    @staticmethod
    def decode(code: int) -> Tuple[int, int, int]:
        return (Morton3D.compact(code), Morton3D.compact(code >> 1), Morton3D.compact(code >> 2))

    @staticmethod
    def get_neighbor(code: int, level: int, dx: int, dy: int, dz: int, max_level: int = 21) -> int:
        x, y, z = Morton3D.decode(code)
        size = 1 << (max_level - level)
        nx, ny, nz = x + dx * size, y + dy * size, z + dz * size
        limit = 1 << max_level
        if (nx < 0 or nx >= limit or ny < 0 or ny >= limit or nz < 0 or nz >= limit):
            return -1
        return Morton3D.encode((nx, ny, nz))
