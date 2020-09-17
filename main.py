# encodin=utf-8

f1 = open('c:/1.txt', 'a+', encoding='utf-8')
with open('./data/raw_data/xiaohuangji50w_nofenci.conv', 'r', encoding='utf-8') as f:
    count = 0
    s = ''
    for line in f:
        if not (line.startswith('E') or line.startswith('M')):
            continue
        if 'E\n' == line:
            count = 0
            if s.endswith('\t'):
                s = s[:-1]
            f1.writelines(s + '\n')
            s = ''
            continue
        count += 1
        if count <= 2:
            tmp = line.replace('\n', '').replace('M ', '').replace('\t', ' ')
            s += (' '.join(tmp.split()) + '\t')
f1.close()