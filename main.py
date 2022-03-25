# Author: Matin Nazamy
# Please see the README for a description of this program.
from doctest import testfile
import os.path
import math


class LanguageModel:
  
    # I/O Class Variables
    outFile = None
    testFileName = ""
    trainFileName = ""

        # Most important Class Variables
    uniGramMap = {}
    biGramMap = {}

        # Training file variables
    totalNumLines= 0
    totalNumberOfTokens = 0
    totalUniqueWords = 0
    totalUniqueBigrams =0

        # Test File variables
    totalNumLinesTestData =0
    totalNumberOfTokensTestData = 0
    totalUniqueWordsTestData =0


    def __init__(self, trainf, testf):

            # assigning i/o variables
        self.trainFileName = trainf
        self.testFileName = testf
        self.outFile = open("outFile.txt","w",encoding="utf8")


            # calling this function to read the training corpus and fill in our word map
        self.startUnigramModel()
        self.processTrainFile() #padding, lowercase, and replacing with <unk>
        # Question 1 and 2 -------------------------------------------

        self.totalUniqueWords = len(self.uniGramMap.keys()) -1
        self.outFile.write("\n1.\nNumber of word types (unique words) in training data is = " + str(self.totalUniqueWords)  )

        self.totalUniqueWords = len(self.uniGramMap.keys()) -1 - self.uniGramMap["<unk>"]
        self.outFile.write("\nNumber of word types after replacing with <unk> is = " +str(self.totalUniqueWords) )

        self.outFile.write("\n\n2.\nTotal Number of tokens in training data is = " + str(self.totalNumberOfTokens )  )


        # Question 3 -------------------------------------------

        percentUnseenUnigramTokens, percentUnseenUnigramTypes = self.processTestFile()
        self.outFile.write("\n\n3.\nThe percentage of unseen words tokens from the test corpus is =   " + str(percentUnseenUnigramTokens)+"%")
        self.outFile.write("\nThe percentage of unseen word types from the test corpus is =   " + str(percentUnseenUnigramTypes)+"%")

        # Question 4 -------------------------------------------

        self.replaceWithUnk()
        percentUnseenBigramTokens, percentUnseenBigramTypes = self.startBigramModel()
        self.outFile.write("\n\n4.\nThe percentage of unseen bigrams tokens from the test corpus is =   " + str(percentUnseenBigramTokens)+"%")
        self.outFile.write("\nThe percentage of unseen bigrams types from the test corpus is =   " + str(percentUnseenBigramTypes)+"%")

        # Question 5 -------------------------------------------

        sample_sentence = "<s> i look forward to hearing your reply . </s>"
        self.outFile.write("\n\n5. The log probability of the sentence '"+sample_sentence+"'  : \n-------------------------------- ")
        self.outFile.write("\n\nUnigrams (max likelihood) =" + str( self.computeLogProb_Unigram(sample_sentence, printStepByStep=True)) + "\n--------------------------------\n") 
        self.outFile.write("\n\nBigrams (max likelihood) = " + str(self.computeLogProb_Bigram_MLE(sample_sentence,printStepByStep=True)) + "\n--------------------------------\n")
        self.outFile.write("\n\nBigrams (add-one smoothing) = " + str(self.computeLogProb_Bigram_Smoothed(sample_sentence,printStepByStep=True))+  "\n--------------------------------\n")


        # Question 6 -------------------------------------------

        self.outFile.write("\n\n6.The perplexity of of the sentence '"+sample_sentence+"' : \n")
        self.outFile.write("\nUnigrams (max likelihood) = " + str(self.computePerplexitySentence_Unigram(sample_sentence)))        
        self.outFile.write("\nBigram (max likelihood) = " + str(self.computePerplexitySentence_Bigram(sample_sentence, smoothed=False)))
        self.outFile.write("\nBigram (smoothed) = " + str(self.computePerplexitySentence_Bigram(sample_sentence, smoothed=True)))

        # Question 7 -------------------------------------------


        self.outFile.write("\n\n7.The perplexity of the entire test corpus :\n")
        self.outFile.write("\nUnigrams (max likelihood) = " + str(self.computePerplexityFile_Unigram()) )
        self.outFile.write("\nBigram (max likelihood) = " +str(self.computePerplexityFile_Bigram(smoothed=False)))
        self.outFile.write("\nBigram (smoothed) = " + str(self.computePerplexityFile_Bigram(smoothed=True)) )

        self.outFile.close()


    # Methods for preprocessing and solving Q1 - Q4 

    def startUnigramModel(self):
        # This method loops through the training file and counts the frequencies of words, storing them in a map.

        with open(self.trainFileName,"r",encoding="utf8") as trainFile:
            for line in trainFile:
                self.totalNumLines +=1
                tokens = line.split()  
                self.totalNumberOfTokens += len(tokens) # Keep track of total number of tokens we've seen excluding <s> and </s>

                for t in tokens: #Reading each word
                    
                    t = t.lower() # convert token to lowercase
                    
                    try:    # keep track of how many times each word occured
                        self.uniGramMap[t] += 1
                    except KeyError:
                        self.uniGramMap[t] = 1


        self.uniGramMap["<s>"] = self.totalNumLines
        self.uniGramMap["</s>"] = self.totalNumLines
        self.uniGramMap["<unk>"] = 0

        self.totalNumberOfTokens += self.totalNumLines  #accounting for one </s> token at the end of every line

    def processTrainFile(self):
        # This function does 4 things in the same loop:
            # Mapping words in the training corpus to <unk>
            # Keeping track of the count of <unk>
            # Seperate and pad sentences with <s> </s>
            # Write everything to a new file "trainPreProcessed.txt"

        with open(self.trainFileName,"r",encoding="utf8") as trainingFile: 
            with open("./trainPreProcessed.txt","w",encoding = "utf8") as processedTrainingFile:

                try:
                    for line in trainingFile:

                        processedTrainingFile.write("<s> ") # adding a new <s> token to the front of the line
                        tokens = line.split()
                        
                        for t in tokens:

                            t = t.lower()

                            if self.uniGramMap[t]<=1:      # Changing singletons tokens to <unk> as necessary
                                t = "<unk>"
                                self.uniGramMap["<unk>"] +=1

                            processedTrainingFile.write(t+ " ")

                        processedTrainingFile.write("</s>\n") # adding a new </s> token to the end of the line

                    
                except KeyError:
                    self.outFile.write("\n********\nthis should never appear, because we've mapped these words already!!")

    def replaceWithUnk(self):
        # After all mapping and prior required questions are complete:
        # Will loop through the unigram map and delete keys that occour only once

        delKeys = []
        for k in self.uniGramMap.keys():
            if self.uniGramMap[k] == 1:
                delKeys.append(k)
        
        for k in delKeys:
            del self.uniGramMap[k]

    def processTestFile(self):
        # This function loops over the test file twice
    


        # 2 things done in this loop
            # 1. Preprocessing the data 
            #          - Padding sentences between <s> </s>
            #          - Converting all words to lowercase
            #          - Replacing unseen test data with <unk>
            #          - Write processed test file to  "testPreProcessed.txt"
            # 2. Compute the percentage of unseen word tokens in the test data

        numUnseenTokens = 0 

        with open(self.testFileName,"r",encoding="utf8") as testFile:
            with open("./testPreProcessed.txt","w",encoding="utf8") as processedTestFile:

                for line in testFile:
                    processedTestFile.write("<s> ")

                    tokens = line.split()
                    self.totalNumLinesTestData +=1
                    self.totalNumberOfTokensTestData += len(tokens) #  keeping track of this number

                    for t in tokens:    #count each token, and write the processed token in the preProcessed file

                        t = t.lower()
                        try:
                            if self.uniGramMap[t] > 1:      # if we've seen this word
                                pass
                            else:       # else we've seen this word exactly once, and should be re-mapped to <unk>
                                t = "<unk>"
                        except: #except if we've never seen this word
                            numUnseenTokens +=1
                            t = "<unk>"

                        processedTestFile.write(t + " ")

                    processedTestFile.write("</s>\n")

        self.totalNumberOfTokensTestData += self.totalNumLinesTestData  # accounting for </s> at end of sentence
        percentUnseenWordTokens = round(100 * numUnseenTokens/(self.totalNumberOfTokensTestData) , 3)  # store  the percentage for unseen word tokens



        # In this loop, just calculating the percent of unseen word TYPES (unique tokens)
        numUnseenWordTypes =0
        readTokens = {}

        with open(self.testFileName,"r",encoding="utf8") as testFile:
            for line in testFile:
                tokens = line.split()

                for t in tokens:

                    t = t.lower()
                    try: 
                        if readTokens[t] >0:    # if we've seen this word type before, skip
                            pass

                    except KeyError:        # exception occours if we did not see that word yet in the test file
                        readTokens[t] = 1           
                        try:
                            if self.uniGramMap[t] > 0:  # now check if this unique word from test  occoured in training data
                                pass
                        
                        except KeyError:         # exception occours if we did not encounter that data in the training data
                            numUnseenWordTypes +=1

        self.totalUniqueWordsTestData = len(readTokens.keys() ) - 1
        percentUnseenWordTypes =  round(100 *numUnseenWordTypes/self.totalUniqueWordsTestData , 3)  # store the percentage for unseen word types

            # Return those two values
        return percentUnseenWordTokens, percentUnseenWordTypes

    def startBigramModel(self):

        #This method does 3 things:
        #   1. Keep counts of all the bigrams from the test data
        #   2. Calculate percent of unseenBigramTokens from test data
        #   3. Calculate percent of unseenBigramTypes from test data

        # -----------------------

        # 1. In this loop, we read in each bigram and keep track of the count within a dictionary of dictionarys. 
 
        with open("./trainPreProcessed.txt","r",encoding="utf8") as train:
            for line in train:
                tokens = line.split()

                for i in range(1,len(tokens)):  
                    prevToken = tokens[i-1]
                    nextToken = tokens[i]
                    try: 
                        submap = self.biGramMap[prevToken]      # tries to see if the first token was ever mapped yet to self.bigramMap
                    except:
                        self.biGramMap[prevToken] = {}  # if this token was never the first part of a bigram, we create that dictionary here
                        submap = self.biGramMap[prevToken] 
                    
                    try:
                        submap[nextToken] +=1       # tries to see if the second token was ever mapped to the dict of the first token
                    except:
                        submap[nextToken] = 1   # initialize the count of this bigram to 1 if its the first time we saw it
                        self.totalUniqueBigrams +=1


        # 2. In this loop, we do calculate what percent of the test file bigrams have never been seen in training
        unseenBigramTokens = 0

        with open("./testPreProcessed.txt") as test:
            for line in test:
                tokens = line.split()

                for i in range(1,len(tokens)):      # nextTokens starts from first word after <s> to </s>
                    prevToken = tokens[i-1]
                    nextToken = tokens[i]
                    try: 
                        if self.biGramMap[prevToken][nextToken] > 0:
                            pass
                    except KeyError:
                        unseenBigramTokens+=1


        # 3. In this loop, we do calculate what percent of the test file bigrams types have never been seen in training

        unseenBigramTypes = 0
        seenBigramTypes = 0
        readBigrams = {}    # we have a local variable dictionary to keep track of the ones weve seen so far

        with open("./testPreProcessed.txt") as test:
            for line in test:
                tokens = line.split()
                for i in range(1,len(tokens)):
                    prevToken = tokens[i-1]
                    nextToken = tokens[i]

                    try:
                        if readBigrams[prevToken][nextToken] > 0:   
                            pass
                    except KeyError:
                        try: 
                            if self.biGramMap[prevToken][nextToken] > 0:
                                seenBigramTypes+=1
                        except KeyError:
                            unseenBigramTypes+=1
                        
                        try:
                            submap = readBigrams[prevToken]
                        except KeyError:
                            readBigrams[prevToken] = {}

                        readBigrams[prevToken][nextToken]=1


        percentUnseenBigramTokens = round((100* unseenBigramTokens)/(self.totalNumberOfTokensTestData),3)
        percentUnseenBigramTypes = round((100* unseenBigramTypes)/(seenBigramTypes+unseenBigramTypes),3)

        return percentUnseenBigramTokens, percentUnseenBigramTypes

    # Methods for Q5 -- computing log probability of a sentence

    def computeLogProb_Unigram(self,sentence: str,printStepByStep: bool):
        # Computes sum of log probabilities of a sentence using Unigram Max-Likelihood model
        sumOfLogs = 0
        tokens = sentence.split()
        for i in range (1,  len(tokens)):
            t = tokens[i]
            try:
                prob = self.uniGramMap[t]/self.totalNumberOfTokens  # prob (w) = count(w) / size of corpus
                logProb = math.log(prob,2) 
            except KeyError:
                prob = self.uniGramMap["<unk>"]/self.totalNumberOfTokens
                logProb = math.log(prob,2)
            sumOfLogs+=logProb
            if printStepByStep:
                self.outFile.write("\nlog( p( " + t + " )  ) = " + str(round(logProb,3)) )
        return sumOfLogs

    def computeLogProb_Bigram_MLE(self,sentence: str,printStepByStep: bool):
        # Computes sum of log probabilities of a sentence using Bigram Max-Likelihood model

        sumOfLogs = 0
        tokens = sentence.split()
        flag = None
        for i in range(1,len(tokens)):
            prevToken = tokens[i-1]
            nextToken = tokens[i]
            try:
                prob = self.biGramMap[prevToken][nextToken] / self.uniGramMap[prevToken]    # prob( w2 | w1) = count(w1  w2) / count(w1) 
                logProb = math.log(prob , 2)
                sumOfLogs+=logProb

                if printStepByStep:
                    self.outFile.write("\nlog( p( " + nextToken + " | " + prevToken + " )  ) = " + str(round(logProb,3)) )

            except KeyError:    # key error if we've never seen this bigram, log ( 0 ) == undefined

                if printStepByStep:
                    self.outFile.write("\nlog( p( " + nextToken + " | " + prevToken + " )  ) = undefined")
                
                if not flag:
                    flag = str("undefined -- Error caused at: log (   p( " + nextToken + " | " + prevToken + " )  )" )  # storing the first error 

        if flag:    # just need this since we print out all parameters but save the first error which caused it 
            return flag 

        return sumOfLogs

    def computeLogProb_Bigram_Smoothed(self,sentence: str,printStepByStep: bool):   
        # Computes sum of log probabilities of a sentence using Bigram Add-One Smoothing model

        sumOfLogs = 0
        tokens = sentence.split()

        for i in range (1,len(tokens)):
            prevToken = tokens[i-1]
            nextToken = tokens[i]
            try: 
                    # prob( w2 | w1) = [ count(w1 w2) + 1 ] /  [ count(w1) + |V| ] 
                prob = (self.biGramMap[prevToken][nextToken] + 1) / (self.uniGramMap[prevToken] + self.totalUniqueWords ) 
            except KeyError:
                prob = 1 / (self.uniGramMap[prevToken] + self.totalUniqueWords )
                
            logProb = math.log(prob, 2)
            sumOfLogs += logProb

            if printStepByStep:
                self.outFile.write("\nlog( p( " + nextToken + " | " + prevToken + " )  ) = " + str(round(logProb,3)) )


        return sumOfLogs

    # Methods for Q6 -- computing perplexity of a sentence

    def computePerplexitySentence_Unigram(self,sentence:str):
        # Computes the perplexity of a sentence using unigram MLE model

        numTokens = len(sentence.split())-1
        avgLog = self.computeLogProb_Unigram(sentence, False)/numTokens
        perplexity = pow(2,-1*avgLog )

        return perplexity
    
    def computePerplexitySentence_Bigram(self,sentence:str,smoothed:bool):
        # Computes the perplexity of a sentence using the specified bigram model

        numTokens = len(sentence.split()) -1
        avgLog =1/numTokens

        if smoothed:        
            avgLog *=self.computeLogProb_Bigram_Smoothed(sentence, False)
        else:
            try:
                avgLog *=self.computeLogProb_Bigram_MLE(sentence, False )
            except TypeError: #get a type error when bigram MLE calculation is undefined
                perplexity = "+inf"
                return perplexity

        perplexity = pow(2,-1*avgLog )
        return perplexity
    
    # Methods for Q7 -- computing perplexity of a file

    def computePerplexityFile_Unigram(self):
        # Computes the perplexity of a file using unigram MLE model

        sumOfLogs =0
        with open("./testPreProcessed.txt","r",encoding="utf8") as testFile:
            for line in testFile:
                sumOfLogs += self.computeLogProb_Unigram(line, False)

        avgLog = sumOfLogs/self.totalNumberOfTokensTestData
 
        try:
            perplexity = pow(2,-avgLog )
        except OverflowError:   # just in case :)
            perplexity = "+inf"
        
        return perplexity

    def computePerplexityFile_Bigram(self, smoothed:bool):
        # Computes the perplexity of a sentence using the specified bigram model

        sumOfLogs = 0        

        with open("./testPreProcessed.txt","r",encoding="utf8") as testFile:
            if smoothed:
                for line in testFile:
                    sumOfLogs += self.computeLogProb_Bigram_Smoothed(line, False)
            else:
                try:
                    for line in testFile:
                        sumOfLogs += self.computeLogProb_Bigram_MLE(line, False)  
                except TypeError:
                    perplexity = "+inf"
                    return perplexity
                
        avgLog  = sumOfLogs/self.totalNumberOfTokensTestData
        
        try:
            perplexity = pow(2,-avgLog)
        except OverflowError:
            self.outFile.write("*********OVERFLOW************") # just in case :)
            perplexity = "+inf"

        return perplexity



def cleanseFileNames():

    ans = input("\nIs your training corpus path == ./train.txt  ?  \t Y/N \t")
    if(ans.lower() == "no" or ans.lower() == "n"):
        trainFileName = input("\nPlease enter the training file name: \t")
        while not os.path.exists(trainFileName):
            trainFileName = input("Couldn't locate file:     " + trainFileName + "\nEnter the name of your training corpus file:            ")
    else:
        trainFileName = "./train.txt"

    ans = input("\nIs your test corpus path == ./test.txt  ?  \t Y/N \t")
    if(ans.lower() == "no" or ans.lower() == "n"):
        testFileName = input("\nPlease enter the test file name: \t")
        while not os.path.exists(testFileName):
            testFileName = input("Couldn't locate file:     " + testFileName + "\nEnter the name of your test corpus file:            ")
    else:
        testFileName = "./test.txt"

    return trainFileName, testFileName


if __name__ == "__main__":

        trainF, testF = cleanseFileNames()
        l = LanguageModel(trainF, testF)

