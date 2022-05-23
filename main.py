import sys


if __name__ == '__main__':
    file_name = sys.argv[1]
    with open(file_name, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    with open(file_name+'small', 'w',encoding='utf-8') as f:
        for i,line in enumerate(lines):
            if i == 1000000:
                break
            f.write(line)
    f.close()