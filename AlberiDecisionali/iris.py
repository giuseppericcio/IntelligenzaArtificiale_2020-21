import treepredict as tp

if __name__ == '__main__':

    print("ESERCIZIO SU IRIS DATASET\n")

    train_data=[]
    test_data=[]

    print("ALL DATASET:\n")
    mydata=tp.aprifile("iris.txt")

    train_data, test_data = tp.splitdataset2(mydata, 60, [])

    print("TRAIN DATA : \n")

    print(str(len(train_data)) + "\n")

    print("TEST DATA: \n")

    print(str(len(test_data)) + "\n")

    iris_tree = tp.buildtree(train_data)

    tp.drawtree(iris_tree, "iris_tree.jpeg")

    print("MISURA PERFORMANCE: \n")

    tp.learningcurve(mydata)

    print("\n")