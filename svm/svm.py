## Spark Application - SVM Programming

import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

def main_spark(sc, trainData, testData, outputFilename):
    # Load and process data
    dataProcessing = sc.textFile(trainData) \
        .map(parseTrainData)
    # Load test data
    testDataLoad = sc.textFile(testData)\
        .map(parseTestData)

    # Build svm model
    #model = SVMWithSGD.train(dataProcessing, iterations=100, step=1.0, regParam=0.01)
	model = SVMWithSGD.train(dataProcessing, iterations=50, step=1.0, regParam=0.01, miniBatchFraction=20.0)

    # Training data
    predict = testDataLoad.map(lambda a: model.predict(a))
    #print(str(loss))

    lable = ' '    

    fp = open(outputFilename, "wb")
    for result in predict:
        if result == 0:
           lable = "business"
        elif result == 1:
           lable = "culture-Arts-Entertainment"
        elif result == 2:
           lable = "computers"
		elif result == 3:
           lable = "education-Science"
		elif result == 4:
           lable = "engineering"
		elif result == 5:
           lable = "health"
		elif result == 6:
           lable = "politics-Society"
		elif result == 7:
           lable = "sports"

        fp.writelines(lable)
        fp.write("\n")
    fp.close()


def parseTrainData(record):
    values = [int(x) for x in record.split(' ')]

    return LabeledPoint(values[0], values[1:])


def parseTestData(record):
    values = [int(x) for x in record.split(' ')]

    return values


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Error: the number of arguments should be three, the outputAddress and the two inputFiles ")
        quit()

    trainData  = sys.argv[1]
    testData = sys.argv[2]
    outputFilename = sys.argv[3]

    # Execute Main functionality
    conf = SparkConf().setAppName("SVM")
    sc = SparkContext(conf=conf)
    main_spark(sc, trainData, testData, outputFilename)
