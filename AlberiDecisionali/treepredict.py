# -*- coding: utf-8 -*-

import random
# Questa funzione verifica se il valore numerico è float
def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

# Questa funzione verifica se il valore numerico è intero
def isint(value):
    try:
        int(value)
        return True
    except:
        return False

# Questa funzione apre un file di testo.txt e lo legge riga per riga
def aprifile(fil="nomefile.txt"):
    data = []
    for line in open(fil):
        srt = line.split(',')
        for count in range(0, len(srt)):
            if (isfloat(srt[count])):
                srt[count] = float(srt[count])
            else:
                srt[count] = srt[count].strip('\n')
        data = data + [srt];
    return data

# Questa funzione scambia i valori di due colonne
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
    f.close();

# Questa funzione divide l'insieme di dati in una parte per il train dell'albero ed un'altra per il test
# e salva l'insieme dei dati in un file con nome v%train.txt dove v è il numero di percentuale di dati da "trainare"  
def splitdataset(data, numdati):
    # print nelement
    tr = []
    te = []
    t = []
    train_number = int((numdati / len(data)) * 100)
    test_number = int(((len(data) - numdati) / len(data)) * 100)

    for i in range(0, numdati):
        t = random.choice(data);
        tr = tr + [t]
        num = data.index(t)
        del data[num]
    f = open(str(train_number) + "%train.txt", "w")
    for row in tr:
        f.write("%s\n" % row)
    f.close();
    
    te = data
    f1 = open(str(test_number) + "%test.txt", "w")
    for row in te:
        f1.write("%s\n" % row)
    f1.close();
    return (tr, te)

# Questa funzione divide l'insieme di dati in una parte per il train dell'albero ed un'altra per il test
# e salva l'insieme dei dati in un file con nome v%train.txt dove v è il numero di percentuale di dati da "trainare"
def splitdataset2(data, numdati, train):
    # print nelement
    tr = []
    te = []
    t = []

    for i in range(0, numdati):
        t = random.choice(data);            #random.choice(data) sceglie casualmente un dato nel dataset diviso per il training
        tr = tr + [t]
        num = data.index(t)
        del data[num]
    v = len(data) - numdati
    f = open(str(v) + "%train.txt", "w")    #salvataggio del file v%train.txt
    if len(train) != 0:
        for row in train:
            f.write("%s\n" % row)
    for row in tr:
        f.write("%s\n" % row)
    f.close();
    
    te = data
    f1 = open(str(v) + "%test.txt", "w")    #salvataggio del file v%test.txt
    for row in te:
        f1.write("%s\n" % row)
    f1.close();
    return (tr, te)

# Questa funzione serve a formattare il file.txt preso da un dataset per costruire l'albero decisionale
# il formato sarà [ x , y , z ]
def mettivirgola(fil="nomefile.txt"):
    l = ""
    f1 = open(fil + "aggiunta.txt", "a")
    with open(fil, "r") as f:
        for line in f:
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace("'", "")
            line = line.strip(' ')
            f1.write(line)

# Creazione della classe nodo decisionale
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col          # colonna del criterio da testare
        self.value = value      # valore che la colonna deve avere per essere vera
        self.results = results  # conserva risultati per il ramo
        self.tb = tb            # prossimo nodo true
        self.fb = fb            # prossimo nodo false


# Divide l'insieme su una colonna specifica. Può gestire valori numerici o valori nominali
def divideset(rows, column, value):  
    # Crea una funzione che ci dice se è presente una riga
    # nel primo gruppo (vero) o nel secondo gruppo (falso)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value  # se i valori sono numerici il
        # criterio vero prende i dati maggiori del value che gli abbiamo passato
    else:
        split_function = lambda row: row[column] == value  # se non sono numerici prende come true
        # il valore dato

    # Divide le righe in due insiemi e li ritorna
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


# Questa funzione mi permette di capire quanti risultati veri
# e quanti risultati falsi ci sono nei set creati sopra
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


# Probabilità che un oggetto posizionato casualmente sarà
# nella categoria sbagliata
def giniimpurity(rows):  # mi dice la purezza del set, se gli oggetti sono tutti omogenei
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
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# L'entropia è la somma di p(x)log2(p(x)) in tutti
# i diversi risultati possibili
def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    # Adesso calcola l'entropia
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def printtree(tree, indent=''):
    # E' questo un nodo foglia?
    if tree.results != None:
        print(str(tree.results))
    else:
        # Stampa il criterio
        print(str(tree.col) + ':' + str(tree.value) + '? ')

        # Stampa i rami
        print(indent + 'T->'),
        printtree(tree.tb, indent + '  ')
        print(indent + 'F->'),
        printtree(tree.fb, indent + '  ')

# Questa funzione restituisce l'ampiezza dell'albero
def getwidth(tree):
    if tree.tb == None and tree.fb == None: return 1
    return getwidth(tree.tb) + getwidth(tree.fb)

# Questa funzione restituisce la profondità dell'albero
def getdepth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


from PIL import Image, ImageDraw

# Questa funzione disegna l'albero e lo salva come immagine JPEG
def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, format="JPEG", quality=40, progessive=True)

# Questa funzione disegna un nodo dell'albero  
def drawnode(draw, tree, x, y):
    if tree.results == None:
        # Prendo l'ampiezza di ogni ramo
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determino lo spazio totale richiesto da questo nodo
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Disegna la stringa della condizione
        # E' POSSIBILE AGGIUNGERE UN font = ImageFont.truetype("arial.ttf", 15) e passare font = font in draw.text (PIU' LEGGIBILE)
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Disegna i collegamenti ai rami
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Disegna i nodi del ramo
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 50, y), txt, (0, 0, 0))


import matplotlib.pyplot as plt

# Questa funzione misura la performance dell'albero decisionale facendo delle prove sul test set
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
    print(len(test))
    # print len(test)
    return percent

# Questa funzione costruisce il grafico di performance dell'albero decisionale
def learningcurve(data):
    testc = data
    percent = 10
    p = []
    perc = []
    t = []
    numdati = (int)((float)(len(testc)) / 100 * percent);
    for i in range(0, 9):
        (train, testc) = splitdataset2(testc, numdati, t)
        t = t + train;
        tree = buildtree(t)
        p = p + [performance(tree, testc)]
        perc = perc + [percent]
        percent = percent + 10;
    line, = plt.plot(perc, p, 'r-')
    plt.xlabel('Training set size')
    plt.ylabel('%' + 'Correct on test set')
    line.set_antialiased(False)
    plt.show()

# Questa funzione ritorna la classificazione(vero o falso) del ramo dell'albero considerato
# in base all'osservazione del valore assunto dal ramo stesso
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

# Questa funzione effettua la potatura dell'albero per diminuire i tempi di calcolo
def prune(tree, mingain):
    # Se i rami non sono foglie, potali
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    # Se entrambi i sottorami sono ora foglie, vedere se
    # dovrebbero essere uniti
    if tree.tb.results != None and tree.fb.results != None:
        # Costruire un dataset combinato
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # Testare la riduzione dell'entropia
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            # Unire i rami
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)

# Questa funzione ritorna la classificazione(vero o falso) "pesata" del ramo dell'albero considerato
# in base all'osservazione del valore assunto dal ramo stesso
def mdclassify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        if v == None:  # classificazione pesata
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items(): result[k] = v * tw
            for k, v in fr.items(): result[k] = v * fw
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

# Questa funzione calcola la varianza di dati per riga
def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance

# Questa funzione costruisce l'albero decisionale a partire dal dataset fornito
def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionnode()
    current_score = scoref(rows)  # scoref in realtà è la funzione dell' entropia

    # Imposta alcune variabili per tracciare i criteri migliori
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Genera l'elenco dei diversi valori in
        # questa colonna
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Ora prova a dividere le righe per ogni valore
        # in questa colonna
        for value in column_values.keys():
            # Per ogni valore nelle colonne che gli diamo lo divide in due set quello che
            # rispetta il criterio e quello che non lo rispetta
            (set1, set2) = divideset(rows, col, value)

            # Guadagno di informazioni
            p = float(len(set1)) / len(rows)  # calcola la probabilità che capiti un risultato vero
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Dopo aver provato tutte le possibili combinazioni per il valore più alto di gain crea
    # due alberi, su cui richiama la funzione di costruzione e crea il nodo se il best_gain
    # è maggiore di 0, perchè se non lo è significa che abbiamo raggiunto la foglia,perchè
    # l' entropia fatta da un set di oggetti uguali è 0 quindi current_score sarà 0 quindi
    # stamperemo il numero di nodi decisione che sono collassati
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))

# Questa funzione crea l'albero decisionale e lo salva come immagine JPEG a partire dal dataset
def createDT(numdati, fil='nomefile.txt'):
    data = aprifile(fil)
    (training, test) = splitdataset(data, numdati, [])
    tree = buildtree(training)
    drawtree(tree, 'K' + str(random.randint(0, 100)) + '-' + fil + '-' + str(numdati) + '-tree.jpg')

# Questa funzione stampa il grafico della perfomance dell'albero decisionale
def performanceMeasure(fil='nomefile.txt'):
    data = aprifile(fil)
    learningcurve(data)
