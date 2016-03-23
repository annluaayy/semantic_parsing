# semantic_parsing
scripts for semantic parsing 

The semantic parsing project is divided into three steps: 
1) lexicon acquisition
2) semantic parser learning
3) sentence generation (though we should consider this step simultaneously before)

  lexicon acquision:
   for this step, two algorithms are initially implemented. Both programs run under the same directory of the one-to-one-mapping-data folder.
   
   = IBM model 1, word-alignment method =
   >>>With EM algorithm integreted into the model for estimating the translation probability
   
   for now, I simply use 10 iterations for em algorithm instead of a smarter convergence way,
   caz this is for detecting how the probabilities change, bigger and bigger for good alignment, smaller for incorrect        alignment
  
  ***input >>> pairs of (NL sentence, MR representation set) (by using InitialData class to read from corpus)***;
  ***output >>> each iteration will product a file.txt, with each line containing a word-alignment pair: (unigram, meaning)           probability***
  
  =  GILL(Graph Intersection Lexicon Learning) =
 >>>An algorithm by David L.Chen in his dissertation of Doctor of Philisophy
  
  By finding the intersections between meaning representations to construct a lexicon for unigram(for now unigrams, but we can     extend to bigrams as well in these two algorithms)
  
  In finding intersections, I use the greedy way(Largest Common Substring Algo), iteratively running the program until no new overlapping can be found. Each time with a score function to reserve K(here 10) highest score element as our potential candidates. Among the K candidates, only the ones who are higher than a certain threshold(here 0.4) win out their way as the meaning of the word.
  
  we can change the k and the threshold parameter in terminal
  
  ***input >>> pairs of (NL sentence, MR representation set) (by using InitialData class to read from corpus)***;
  ***output >>> a file.txt(like a lexicon), with each line containing an unigram and its best meaning***
  
   
