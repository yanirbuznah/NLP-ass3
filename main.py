def read_file(file_path):
    words = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        while len(lines) > 0:
            words[lines[0].strip()] = []
            for i,line in enumerate(lines[1:]):
                if line.strip() == '*********':
                    lines = lines[i+2:]
                    break
                words[lines[0].strip()].append(line.strip())
    return words



def combine_files(file_paths, output_file):
    lists = [read_file(files) for files in file_paths]
    words = lists[0].keys()
    with open(output_file, 'w',encoding='utf-8') as f:
        for i,word in enumerate(words):
            f.write(f"{word}\n")
            for x,y,z in zip(lists[0][word],lists[1][word],lists[2][word]):
                f.write(f"{x} {y} {z}\n")
            f.write("*********\n")



def precision(retrives,k = 20):
    precisions = []
    for i in range(k):
        x = i - len([1 for j in retrives if j <=i]) + 1
        precisions.append(x/(i+1))
    return precisions

def precision_from_words(retrives,gold):
    res = 0
    for ret in retrives:
        if ret in gold:
            res += 1
    return res/len(retrives)

car_top20 ='truck vehicle driver vehicle motor racing ford driver motor formula drive racing racing race automobile truck truck motor automobile motorcycle race chevrolet formula drive auto traffic bicycle toyota carriage gt boat motorcycle formula high-speed trailer nascar horse bus race terminal motorcycle bmw locomotive wagon bus bmw bmw auto carriage crash ambulance auto speed chevrolet'


car_top20_dep = 'driver racing motor service race truck motorcycle formula traffic carriage boat high-speed horse terminal bmw bus auto ambulance chevrolet stock'
car_top20_sentence = 'vehicle motor driver drive racing truck automobile chevrolet auto toyota gt formula nascar race motorcycle wagon bmw crash speed drunk'
car_top20_window = 'truck vehicle ford formula racing automobile motor race drive bicycle stock motorcycle trailer bus prototype locomotive bmw carriage auto toy'
if __name__ == '__main__':
# combine_files(['top20contexts_2.txt', 'top20contexts_full.txt', 'top20contexts_dep.txt'], 'top20contexts.txt')
    x = precision([19,12])
    car_gold = set(car_top20.split())
    car_window_retrive = car_top20_window.split()
    AP_car_window = sum([precision_from_words(car_window_retrive[:i + 1], car_gold) for i in range(20)]) / len(car_gold)
    print(AP_car_window)

    car_sentence_retrive = car_top20_sentence.split()
    AP_car_sentence = sum([precision_from_words(car_sentence_retrive[:i + 1], car_gold) for i in range(20)]) / len(car_gold)
    print(AP_car_sentence)

    car_dep_retrive = car_top20_dep.split()
    AP_car_dep = sum([precision_from_words(car_dep_retrive[:i+1],car_gold) for i in range(20)])/len(car_gold)
    print(AP_car_dep)


    print(len(car_gold))

