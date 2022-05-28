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

def precision_from_words(retrived,gold):
    res = 0
    for ret in retrived:
        if ret in gold:
            res += 1
    return res/len(retrived)


def calc_precision(results, gold, count=20):
    retrived = results.split()
    AP = sum([precision_from_words(retrived[:i + 1], gold) for i in range(count) if retrived[i] in gold]) / len(gold)
    return AP

def calc_AP_context(top20_window, top20_sentence, top20_dep, gold):
    AP_window = calc_precision(top20_window, gold)
    print(f"AP_window: {AP_window}")

    AP_sentence = calc_precision(top20_sentence, gold)
    print(f"AP_sentence: {AP_sentence}")    

    AP_dep = calc_precision(top20_dep, gold)
    print(f"AP_dep: {AP_dep}")

def context_base_AP():
    car_top20_topical ='truck vehicle driver vehicle motor racing ford driver motor formula drive racing racing race automobile truck truck motor automobile motorcycle race chevrolet formula drive auto traffic bicycle toyota carriage gt boat motorcycle formula high-speed trailer nascar horse bus race terminal motorcycle bmw locomotive wagon bus bmw bmw auto carriage crash ambulance auto speed chevrolet'
    car_top20_semantical ='truck vehicle vehicle ford motor automobile truck truck motor automobile motorcycle chevrolet auto bicycle toyota carriage boat motorcycle trailer horse bus motorcycle bmw locomotive wagon bus bmw bmw auto carriage ambulance auto chevrolet'
    car_top20_dep = 'driver racing motor service race truck motorcycle formula traffic carriage boat high-speed horse terminal bmw bus auto ambulance chevrolet stock'
    car_top20_sentence = 'vehicle motor driver drive racing truck automobile chevrolet auto toyota gt formula nascar race motorcycle wagon bmw crash speed drunk'
    car_top20_window = 'truck vehicle ford formula racing automobile motor race drive bicycle stock motorcycle trailer bus prototype locomotive bmw carriage auto toy'

    piano_top20_topical = 'flute cello cello cello flute flute op concerto concerto concerto sonata sonata sonata op viola guitar viola bass viola quartet guitar trumpet percussion op solo instrument trumpet saxophone guitar solo bass trumpet percussion keyboard trio horn percussion composition saxophone soloist saxophone keyboard string string alto drum violinist quartet organ bass soloist ensemble keyboard orchestra pianist soprano acoustic acoustic tenor'
    piano_top20_semantical = 'flute cello cello cello flute flute viola guitar viola bass viola guitar trumpet percussion instrument trumpet saxophone guitar bass trumpet percussion keyboard trio horn percussion saxophone saxophone keyboard string string drum organ bass keyboard'
    piano_top20_dep =  'cello flute concerto sonata viola bass guitar op trumpet solo percussion horn saxophone keyboard alto quartet soloist lesson soprano tenor'
    piano_top20_sentence = 'cello flute concerto sonata op viola quartet percussion instrument guitar trumpet trio composition saxophone string violinist bass keyboard pianist acoustic'
    piano_top20_window = 'flute cello op concerto sonata guitar viola trumpet solo saxophone bass keyboard percussion soloist string drum organ ensemble orchestra acoustic'
    
    car_gold_topical = set(car_top20_topical.split())
    car_gold_semantical = set(car_top20_semantical.split())
    piano_gold_topical = set(piano_top20_topical.split())
    piano_gold_semantical = set(piano_top20_semantical.split())

    print("car topical:")
    calc_AP_context(car_top20_window, car_top20_sentence, car_top20_dep, car_gold_topical)
    print("car semantical:")
    calc_AP_context(car_top20_window, car_top20_sentence, car_top20_dep, car_gold_semantical)
    
    print("piano topical:")
    calc_AP_context(piano_top20_window, piano_top20_sentence, piano_top20_dep, piano_gold_topical)
    print("piano semantical:")
    calc_AP_context(piano_top20_window, piano_top20_sentence, piano_top20_dep, piano_gold_semantical)

def calc_AP_word_2_vec(top20_bow_5, top20_dep, gold):
    AP_window = calc_precision(top20_bow_5, gold)
    print(f"AP_bow_5: {AP_window}")  

    AP_dep = calc_precision(top20_dep, gold)
    print(f"AP_dep: {AP_dep}")
    
def word_2_vec_AP():
    car_top20_topical ='cars truck truck suv automobile	vehicle vehicle	minivan motorbike cars motorcycle speedboat driver racecar minivan automobile suv motorcar lorry jeep motorcar limousine mid-engined minibus limousine	lorry front-engined	limo moped	motorcycle motorhome bike mercedes-benz motorhome bike taxicab rear-engined roadster three-wheeled wagon'     
    car_top20_semantical = 'cars truck truck suv automobile	vehicle vehicle	minivan motorbike cars motorcycle speedboat racecar minivan automobile suv motorcar lorry jeep motorcar limousine minibus limousine	lorry limo moped	motorcycle motorhome bike mercedes-benz motorhome bike taxicab roadster wagon'     
    car_top20_dep = 'truck suv vehicle minivan cars speedboat racecar automobile motorcar jeep limousine minibus lorry limo motorcycle bike motorhome taxicab roadster wagon'
    car_top20_bow_5 = 'cars truck automobile vehicle motorbike motorcycle driver minivan suv lorry motorcar mid-engined limousine front-engined moped motorhome mercedes-benz bike rear-engined three-wheeled'

    piano_top20_topical = 'violin violin cello cello harpsichord harpsichord clarinet saxophone viola clarinet flute guitar bassoon trombone violoncello mandolin oboe vibraphone concerto marimba saxophone accordion accordion pianoforte harp bassoon trombone fortepiano sonatas violoncello trumpet trumpet mandolin harmonica pianoforte clavinet vibraphone clavichord concertos euphonium'
    piano_top20_semantical = 'violin violin cello cello harpsichord harpsichord clarinet saxophone viola clarinet flute guitar bassoon trombone violoncello mandolin oboe vibraphone marimba saxophone accordion accordion pianoforte harp bassoon trombone fortepiano violoncello trumpet trumpet mandolin harmonica pianoforte clavinet vibraphone clavichord euphonium'
    piano_top20_dep =  'violin cello harpsichord saxophone clarinet guitar trombone mandolin vibraphone marimba accordion pianoforte bassoon fortepiano violoncello trumpet harmonica clavinet clavichord euphonium'
    piano_top20_bow_5 = 'violin cello harpsichord clarinet viola flute bassoon violoncello oboe concerto saxophone accordion harp trombone sonatas trumpet mandolin pianoforte vibraphone concertos'

    car_gold_topical = set(car_top20_topical.split())
    car_gold_semantical = set(car_top20_semantical.split())
    piano_gold_topical = set(piano_top20_topical.split())
    piano_gold_semantical = set(piano_top20_semantical.split())

    print("car topical:")
    calc_AP_word_2_vec(car_top20_bow_5, car_top20_dep, car_gold_topical)
    print("car semantical:")
    calc_AP_word_2_vec(car_top20_bow_5, car_top20_dep, car_gold_semantical)
    
    print("piano topical:")
    calc_AP_word_2_vec(piano_top20_bow_5, piano_top20_dep, piano_gold_topical)
    print("piano semantical:")
    calc_AP_word_2_vec(piano_top20_bow_5, piano_top20_dep, piano_gold_semantical)


def main():
    context_base_AP()
    word_2_vec_AP()


if __name__ == '__main__':
    main()


