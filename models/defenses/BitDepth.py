class BitDepthReduction(object):
    def __init__(self, compressed_bit=4):
        self.compressed_bit = compressed_bit
      

    def __call__(self, xs):
        bits = 2 ** self.compressed_bit #2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = xs_255 / 255

        return xs_compress