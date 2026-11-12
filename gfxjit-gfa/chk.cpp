#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
static inline void fp6_pack32(const uint8_t *src, uint8_t *dst) {
  for (int i = 0; i < 24; ++i) dst[i] = 0;
  for (int i = 0; i < 32; ++i) {
    int bit = i * 6;
    uint8_t v = src[i] & 0x3F;
    dst[bit >> 3] |= (uint8_t)(v << (bit & 7));
    if ((bit & 7) > 2) dst[(bit >> 3) + 1] |= (uint8_t)(v >> (8 - (bit & 7)));
  }
}
int main() {
  const int M = 256, K = 512, kBM = 256, kBK = 128, kC1Base = 16384;
  const int ksteps = K / kBK;
  std::vector<uint8_t> An((size_t)M * K);
  for (size_t i = 0; i < An.size(); ++i) An[i] = (uint8_t)(i * 7 + 3) & 0x3F;
  std::vector<uint8_t> dst((size_t)(M / kBM) * ksteps * 24576, 0);
  for (int b = 0; b < M / kBM; ++b)
    for (int ks = 0; ks < ksteps; ++ks)
      for (int t = 0; t < 16; ++t)
        for (int lane = 0; lane < 64; ++lane) {
          uint8_t *base = &dst[((size_t)b * ksteps + ks) * 24576];
          const int o = 64 * t + lane;
          const int row = b * kBM + t * 16 + lane % 16;
          const int kbase = ks * kBK + (lane / 16) * 32;
          uint8_t frag[24];
          fp6_pack32(&An[(size_t)row * K + kbase], frag);
          memcpy(base + 16 * o, frag, 16);
          memcpy(base + kC1Base + 8 * o, frag + 16, 8);
        }
  int bad = 0;
  for (int ks = 0; ks < ksteps; ++ks) {
    const uint8_t *stage = &dst[(size_t)ks * 24576];
    for (int wm = 0; wm < 2; ++wm)
      for (int i = 0; i < 8; ++i)
        for (int lane = 0; lane < 64; ++lane) {
          uint8_t got[24];
          memcpy(got, stage + wm * 8192 + lane * 16 + i * 1024, 16);
          memcpy(got + 16, stage + kC1Base + wm * 4096 + lane * 8 + i * 512, 8);
          uint8_t want[24];
          const int row = 128 * wm + 16 * i + lane % 16;
          fp6_pack32(&An[(size_t)row * K + ks * kBK + (lane / 16) * 32], want);
          if (memcmp(got, want, 24)) {
            if (bad < 3) printf("MISMATCH ks=%d wm=%d i=%d lane=%d\n", ks, wm, i, lane);
            ++bad;
          }
        }
  }
  printf(bad ? "BAD %d\n" : "OK\n", bad);
}
