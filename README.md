# intradomain_toolkit
The toolkit is a first experimental release based on the paper "Identification of Intra-Domain Ambiguity Using Transformer Based Learning". The mentioned package can be installed and used for the contextual disambiguation of a natural language corpus (English).

# INTRADOMAIN AMBIGUITY

DISAMBIGUATE INTRADOMAIN CONTEXTUAL AMBIGUITY

## Instructions

1. Install:

```
pip install intradomain_toolkit
```

2. Disambiguate a certain corpus:

```python

from intradomain_toolkit import disambiguation

# initialize context object (to disambiguate a corpus)
contextual = disambiguation.disambiguate()
```
# After calling the contextual object, the following arguments will be displayed :- 
  
  1) Enter the number of target terms that you wish to disambiguate:- 

  The user shall mention the number of terms to be disambiguated (integer type input only)
  
  2) Enter the target term:- 
  
  The user shall input the target terms to be studied (string input)
  
  3) Enter input for a starting label for which the threshold plot is to be obtained:- 
  
  Threshold Plots look like this!
  ![command_1_cluster0__label_14_scatter](https://user-images.githubusercontent.com/44119212/126865235-e572e187-22d6-4e51-83c5-f1e63c082803.png)
  
  (Threshold Plot for the word Command in the Mixed Corpus)

  to obtain a plot for a particular word starting from a particular label number of that word, the user shall input asked starting label.
  
  4) Enter input for a ending label for which the threshold plot is to be obtained:-
  
  This is the end limit for the threshold plot. The loop will end at this number starting from argument 3. For example a user enters 12 in argument 3 (Enter         input for a starting label for which the threshold plot is to be obtained) and 16 in argument 4 (Enter input for a ending label for which the threshold plot is   to be obtained), then the user will get 4 threshold plots for a particular cluster (argument 5).
  
  5) Enter the cluster number for which the threshold plot is to be obtained:-
 
  The user shall enter the cluster for which the threshold plots are to be obtained. Note that the algorithm will take the maximum input not greater than the       number of optimal_k. For eg if the optimal_k is three, then the user can input 
    a) 0 for obtaining the threshold plot for cluster 1
    b) 1 for obtaining the threshold plot for cluster 2
    c) 2 for obtaining the threshold plot for cluster 3
  
  6) Enter the threshold for the context words to be obtained:-
  
  The user shall enter the threshold for the cosine similarity values in order to obtain the context words above the mentioned threshold for a particular target     term. The suggested experimental threshold is 0.45
  
  7) Enter your path for the text corpus:-
  
  The user shall enter the path for the text corpus
  
  8) Enter the path where you wish to save all the plots:-

  The user shall enter the path where he/she wishes to save the obtained results.
