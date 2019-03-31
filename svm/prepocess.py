## Spark Application - SVM Programming

try:
    from pyspark import SparkConf, SparkContext
except:
    print
    "No spark libraries found, ensure you are running locally."

import sys

# Global variables.
wordList_ = []

def main_spark(sc, trainData, testData, outputTrain, outputTest):
    # Load and process data
    wordList = sc.textFile(trainData) \
        .flatMap(allWords)\
        .distinct()\
        .collect()
    
    #print(wordList)
    
    global wordList_
    wordList_ = sc.broadcast(wordList)

    # preprocess train data
    dataProcessing = sc.textFile(trainData)\
        .map(parseTrainData)\
        .collect()
    #print(dataProcessing)
    
    fp = open(outputTrain, "wb")
    for result in dataProcessing:
        fp.write(str(result[0]))
        fp.write(" ")
        toStr = " "
        for i in result[1]:
            toStr.join(i)
        fp.writelines(toStr)
        fp.write("\n")
    fp.close()
    
    # preprocess test data
    testDataProcessing = sc.textFile(testData)\
        .map(parseTesetData)\
        .collect()

    fp = open(outputTest, "wb")
    for result in testDataProcessing:
        toStr = " "
        for i in result:
            toStr.join(i)
        fp.writelines(toStr)
        fp.write("\n")
    fp.close()


def allWords(record):
    attributes = record.split(' ')[0:-1]
    return attributes


def parseTrainData(record):
    lableTmp = record.split(' ')[-1]
    lable = '0'
    if lableTmp == "business":
        lable = '0'
    elif lableTmp == "computers":
        lable = '1'
    elif lableTmp == "culture-Arts-Entertainment":
        lable = '2'
    elif lableTmp == "education-Science":
        lable = '3'
    elif lableTmp == "engineering":
        lable = '4'
    elif lableTmp == "health":
        lable = '5'
    elif lableTmp == "politics-Society":
        lable = '6'
    elif lableTmp == "sports":
        lable = '7'


    size = len(wordList_.value)
    attributesData = [0 for i in range(size)]
    attributes = record.split(' ')[0:-1]
    for word in attributes:
        position = wordList_.value.index(word)
        attributesData[position] += 1

    return (lable, attributesData)


def parseTestData(record):
    size = len(wordList_.value)
    attributesData = [0 for i in range(size)]
    attributes = record.split(' ')[0:-1]
    for word in attributes:
        position = wordList_.value.index(word)
        attributesData[position] += 1

    return attributesData


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Error: the number of arguments should be three, the outputAddress and the two inputFiles ")
        quit()

    trainData  = sys.argv[1]
    testData = sys.argv[2]
    outputTrain = sys.argv[3]
    outputTest = sys.argv[4]

    # Execute Main functionality
    conf = SparkConf().setAppName("SVM_Prepocess")
    sc = SparkContext(conf=conf)
    main_spark(sc, trainData, testData, outputTrain, outputTest)
