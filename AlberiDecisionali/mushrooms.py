import treepredict as tp

if __name__ == '__main__':

    print("ESERCIZIO SU MUSHROOMS DATASET\n")

    train_data=[]
    test_data=[]

    print("ALL DATASET:\n")
    mydata=tp.aprifile("mushrooms_final.txt")

    train_data, test_data = tp.splitdataset2(mydata, 4875, [])

    print("TRAIN DATA : \n")

    print(str(len(train_data)) + "\n")

    print("TEST DATA: \n")

    print(str(len(test_data)) + "\n")

    mushrooms_tree = tp.buildtree(train_data)

    tp.drawtree(mushrooms_tree, "mushrooms_tree.jpeg")

    print("MISURA PERFORMANCE: \n")

    tp.learningcurve(mydata)

    print("\n")