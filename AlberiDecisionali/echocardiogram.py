import treepredict as tp

if __name__ == '__main__':

    print("ESERCIZIO SU ECHOCARDIOGRAM DATASET\n")

    train_data=[]
    test_data=[]

    print("ALL DATASET:\n")
    mydata=tp.aprifile("echocardiogram.txt")

    train_data, test_data = tp.splitdataset2(mydata, 105, [])

    print("TRAIN DATA : \n")

    print(str(len(train_data)) + "\n")

    print("TEST DATA: \n")

    print(str(len(test_data)) + "\n")

    echocardiogram_tree = tp.buildtree(train_data)

    tp.drawtree(echocardiogram_tree, "echocardiogram_tree.jpeg")

    print("MISURA PERFORMANCE: \n")
    
    tp.learningcurve(mydata)

    print("\n")