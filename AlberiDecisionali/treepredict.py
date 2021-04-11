# -*- coding: utf-8 -*-
# my_data=[['slashdot','USA','yes',18,'None'],
#           ['google','France','yes',23,'Premium'],
#        ['digg','USA','yes',24,'Basic'],
#        ['kiwitobes','France','yes',23,'Basic'],
#        ['google','UK','no',21,'Premium'],
#       ['(direct)','New Zealand','no',12,'None'],
#        ['(direct)','UK','no',21,'Basic'],
#        ['google','USA','no',24,'Premium'],
#        ['slashdot','France','yes',19,'None'],
#        ['digg','USA','no',18,'None'],
#       ['google','UK','no',18,'None'],
#        ['kiwitobes','UK','no',19,'None'],
#        ['digg','New Zealand','yes',12,'Basic'],
#        ['slashdot','UK','no',21,'None'],
#        ['google','UK','yes',18,'Basic'],
#        ['kiwitobes','France','yes',19,'Basic']]
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def isint(value):
    try:
        int(value)
        return True
    except:
        return False


# train_data=[]
# test_data=[]
def aprifile(fil="nomefile.txt"):
    data = []
    for line in file(fil):
        srt = line.split(",")
        for count in range(0, len(srt)):
            if isfloat(srt[count]):
                srt[count] = float(srt[count])
            else:
                srt[count] = srt[count].strip("\n")
        data = data + [srt]
    return data


# train_data=aprifile('iristraining 40%.txtaggiunta.txt')
# test_data=aprifile('iristest 60%.txtaggiunta.txt')
d = aprifile("mushroom.txt")


def cambiacolonna(data):
    for i in data:
        for l in range(0, len(i)):
            if l == 0:
                var = i[l]
                i[l] = i[22]
                i[22] = var
    f = open("prova.txt", "w")
    for row in data:
        f.write("%s\n" % row)
    f.close()


def createdataset(data, numdati):
    # print nelement
    tr = []
    te = []
    t = []
    for i in range(0, numdati):
        t = random.choice(data)
        tr = tr + [t]
        num = data.index(t)
        del data[num]
    v = len(data) - numdati
    f = open(str(v) + "%train.txt", "w")
    for row in tr:
        f.write("%s\n" % row)
    f.close()
    # for dato in data:
    # if dato not in tr:
    #  if dato
    # print dato
    te = data
    f1 = open(str(v) + "%test.txt", "w")
    for row in te:
        f1.write("%s\n" % row)
    f1.close()
    return (tr, te)


def createdataset(data, numdati, train):
    # print nelement
    tr = []
    te = []
    t = []
    for i in range(0, numdati):
        t = random.choice(data)
        tr = tr + [t]
        num = data.index(t)
        del data[num]
    v = len(data) - numdati
    f = open(str(v) + "%train.txt", "w")
    if len(train) is not 0:
        for row in train:
            f.write("%s\n" % row)
    for row in tr:
        f.write("%s\n" % row)
    f.close()
    # for dato in data:
    # if dato not in tr:
    #  if dato
    # print dato
    te = data
    f1 = open(str(v) + "%test.txt", "w")
    for row in te:
        f1.write("%s\n" % row)
    f1.close()
    return (tr, te)


def mettivirgola(fil="nomefile.txt"):
    l = ""
    f1 = open(fil + "aggiunta.txt", "a")
    with open(fil, "r") as f:
        for line in f:
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("'", "")
            line = line.strip(" ")
            f1.write(line)


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # colonna del criterio da testare
        self.value = value  # valore che la colonna deve avere per essere vera
        self.results = results  # conserva risultati per il ramo
        self.tb = tb  # prossimo nodo true
        self.fb = fb  # prossimo nodo false


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows, column, value):  # divide le righe in due set di dati in base
    # alla specifica colonna uno che rispecchia value(valore della colonna l' altro no
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = (
            lambda row: row[column] >= value
        )  # se i valori sono numerici il
        # criterio vero prende i dati maggiori del value che gli abbiamo passato
    else:
        split_function = (
            lambda row: row[column] == value
        )  # se non sono numerici prende come true
        # il valore dato

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


# Create counts of possible results (the last column of
# each row is the result)
def uniquecounts(rows):  # questa funzione mi permette di capire quando risultati veri
    # e falsi ci sono nei set creati sopra
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# Probability that a randomly placed item will
# be in the wrong category
def giniimpurity(
    rows,
):  # mi dice la purezza del set, se gli oggetti sono tutti omogenei
    # allora non ho possibilità d' errore, altrimenti se invece ho due possibili risultati
    # set l' impurità e del 50% (con set di 2 elementi), in altri termini vede la probabili
    # tà con cui cade un evento nel set poi la va a moltiplicare per le altre probabilità
    # di oggetti diverse dalle sue e somma i valori
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
    from math import log

    def log2(x): return log(x) / log(2)
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def printtree(tree, indent=""):
    # Is this a leaf node?
    if tree.results != None:
        print str(tree.results)
    else:
        # Print the criteria
        print str(tree.col) + ":" + str(tree.value) + "? "

        # Print the branches
        print indent + "T->",
        printtree(tree.tb, indent + "  ")
        print indent + "F->",
        printtree(tree.fb, indent + "  ")


def getwidth(tree):
    if tree.tb == None and tree.fb == None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb == None and tree.fb == None:
        return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawtree(tree, jpeg="tree.jpg"):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, format="JPEG", quality=40, progessive=True)


def drawnode(draw, tree, x, y):
    if tree.results == None:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) +
                  ":" + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = " \n".join(["%s:%d" % v for v in tree.results.items()])
        draw.text((x - 50, y), txt, (0, 0, 0))


def performance(tree, test):
    # t=[]
    # f=[]
    t = 0
    for row in test:
        # if(isint(classify(row,tree))):
        results = classify(row, tree)
        for r in results:
            if r == row[len(row) - 1]:
                t = t + 1
                # t=t+['true']
            # else:
            # f=f+['false']
    percent = float(t) / len(test)
    print len(test)
    # print len(test)
    return percent


def fperformance(data):
    testc = data
    percent = 10
    p = []
    perc = []
    t = []
    numdati = (int)((float)(len(testc)) / 100 * percent)
    for i in range(0, 5):
        (train, testc) = createdataset(testc, numdati, t)
        t = t + train
        tree = buildtree(t)
        p = p + [performance(tree, testc)]
        perc = perc + [percent]
        percent = percent + 10
    (line,) = plt.plot(perc, p, "r-")
    plt.xlabel("percentuale dati training")
    plt.ylabel("percentuale successi")
    line.set_antialiased(False)
    plt.show()


def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


def prune(tree, mingain):
    # If the branches aren't leaves, then prune them
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results != None and tree.fb.results != None:
        # Build a combined dataset
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


def mdclassify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        if v == None:  # classificazione pesata
            tr, fr = mdclassify(observation, tree.tb), mdclassify(
                observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = v * fw
            return result
        else:  # semplice classificazione perchè il valore è presente
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


def variance(rows):
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


def buildtree(rows, scoref=entropy):
    if len(rows) == 0:
        return decisionnode()
    # scoref in realtà è la funzione dell' entropia
    current_score = scoref(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in column_values.keys():
            # per ogni valore nelle colonne che gli diamo lo divide in due set quello che rispe
            # tta il criterio e quello no
            (set1, set2) = divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(
                rows
            )  # calcola la propabilità che capiti un risultato vero
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Create the sub branches
    # dopo aver provato tutte le possibili combinazioni per il valore più alto di gain crea
    # due alberi, su cui richiama la funzione di costruzione e crea il nodo se il best_gain
    # è maggiore di 0, perchè se non lo è significa che abbiamo raggiunto la foglia,perchè
    # l' entropia fatta da un set di oggetti uguali è 0 quindi current_score sarà 0 quindi
    # stamperemo il numero di nodi decisione che sono collassati
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(
            col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch
        )
    else:
        return decisionnode(results=uniquecounts(rows))


def createDT(numdati, fil="nomefile.txt"):
    data = aprifile(fil)
    (training, test) = createdataset(data, numdati, [])
    tree = buildtree(training)
    drawtree(
        tree,
        "K"
        + str(random.randint(0, 100))
        + "-"
        + fil
        + "-"
        + str(numdati)
        + "-tree.jpg",
    )


def performanceMeasure(fil="nomefile.txt"):
    data = aprifile(fil)
    fperformance(data)


r = createDT(30, "mushroom.txt")

g = performanceMeasure("mushroom.txt")
