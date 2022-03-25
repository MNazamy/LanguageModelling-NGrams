# Language Modelling using N-Grams

## Instructions
- Pre-requisites:
    - Have training corpus and test corpus directories available
    - have *main.py* downloaded 
- Open a terminal in the same directory as *main.py* and run the the command **python main.py**
- When prompted, enter "Y" if your train file directory matches the prompt
    - If your answer is "N", then specify the directory of the training file
- When prompted, enter "Y" if your test file directory matches the prompt
    - If your answer is "N", then specify the directory of the test file

## Overview
- This program *main.py* will create the following language models
    - Unigram Language Model (Maximum Likelihood Estimate)
    - Bigram Language Model (Maximum Likelihood Estimate)
    - Bigram Language Model (Add-One Smoothing)
- After creating the models, it will evaluate the questions addressed below and write them to a file called **"outFile.txt"**. 

## PreProcessing
- Words that appear once in the training corpus will be replaced with "UNKNOWN" token
- Words that appear in the test corpus but never appear in the training corpus will be treated as the "UNKNOWN" token
- All sentences in training and test corpora will be padded between "START" and "STOP" tokens
- All tokens in training and test corpora will be made into lowercase 

## This program will answer the following questions

1. How many word types (unique words) are there in the training corpus? (including the "STOP" token, the "UNKNOWN" token, and not including the "START" token )

2. How many tokens are there in the training corpus after pre processing? (including the "STOP" token, the "UNKNOWN" token, and not including the "START" token )

3. What percentage of word tokens and word types in the test corpus did not occur in training? (before mapping training and test corpora words to the "UNKNOWN" token)

4. What percentage of bigram (tokens and types) from the test corpus appears in the training corpus. (after mapping training and test corpora words to the "UNKNOWN" token)

5. Under the 3 models, what is the log probability of the following sentence   *"I look forward to hearing your reply."*

6. Under the 3 models, what is the perplexity of the following sentence    *"I look forward to hearing your reply."*

7. Under each the 3 models, what is the perplexity of the entire test corpus? 
