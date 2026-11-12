import sys
src = sys.argv[1]
out = sys.argv[2]
base = open('/home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa/%s' % src).read()
a = """  for (auto &x : SA) x = 124 + (rand() % 7);"""
# A[row][k] = 1,2,3,4 by row%4 (codes 8,16,20,24); B all 1.0; scales 1.0
code = """  { static const uint8_t vcode[4] = {8, 16, 20, 24};
    for (int r = 0; r < s.M; ++r)
      for (int k = 0; k < s.K; ++k) An[(size_t)r * s.K + k] = vcode[r % 4];
    for (auto &x : Bn) x = 8; }
"""
s = base.replace(a, code + a)
s = s.replace("""  for (auto &x : SA) x = 124 + (rand() % 7);""",
              """  for (auto &x : SA) x = 127;""")
s = s.replace("""  for (auto &x : SB) x = 124 + (rand() % 7);""",
              """  for (auto &x : SB) x = 127;""")
s = s.replace('''    printf("max rel err''',
              '''    { printf("PROBE C[m][0] for m=0..15 (expect 512,1024,1536,2048 repeating):\\n ");
      for (int m = 0; m < 16; ++m) printf(" %g", bf16_decode(C[(size_t)m * s.N]));
      printf("\\n"); }
    printf("max rel err''')
open('/home/kerrwang/.claude/jobs/89e9b96e/tmp/gfa/%s' % out, 'w').write(s)
print("ok")
