import sys


if len(sys.argv) >= 1:
    lines = file(sys.argv[1]).readlines()
else:
    lines = file('sub.txt').readlines()

if len(sys.argv) >= 2:
    fd = open(sys.argv[2], 'w')
else:
    fd = open('cvt.txt', 'w')

for line in lines:
    tokens = line.strip().split(' ')
    fd.write(tokens[0].strip())

    cnt = 1
    for token in tokens[1:]:
        idx, data = tuple(token.strip().split(':'))
        
        while cnt < int(idx):
            fd.write(' 0.0')
            cnt += 1

        fd.write(' ' + data)
        cnt += 1
    fd.write('\n')

fd.close()
