# AbstractRNN

**Character Level**

For the character level RNN, I adapted this piece of code: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

I experimented with different sequence lengths but ended up choosing a length of 20 because it seemed to work fine and the longer lengths were much slower without much of a benefit. 

In order to generate new characters for the sequence, I used the model probability predictions to draw a single character from a multinomial distribution. I found that using a 'temperature' of 0.8 ended up giving me a good balance of diversity in word choice as well as proper punctuation. 

I inserted a character token that was not in the dataset to signify the end of an abstract but the model never learned to predict this particular token, so I simply generated 1000 characters every time. Therefore, the abstracts typically end in the middle of a sentence. 

To generate abstracts, I used the first 20 characters of an actual abstract as the prompt and then let the model generate the next 1000 characters. 

Here is an example abstract that was produced:

----- Generating with seed: "Long-term depression"

"Long-term depression. The periposary after the non-classion the adpless of �dama enroding indicates that the position of nerve ligand in the hippocampal cortex, HI1 modifies. The structures of the NMDA. In the direct properties of Slave model in extracellular neurons during by caining ollable describes oversal normal induction. Escopolamine }-protein activity in the group and ither the frequency in the rats in the intracellular fiber and functional actions were developed in or free treatments of opioural processess in the similation of spive micrological factor male and attenuated neurovalnetic actions and the strial vascular responses of interaction of a sungies of rat against rats of #-Lyphamia, suggesting that which soges of the dentate gyrus responses were experimental in the present time intermed with lest than �gruction and neurodesensity in the short-term potentiation which synapses, but for exorcis of the noreonal disruption and antibodies is studied in �dentred than �reactic action. �hockty was a"

As you can see, the model sometimes abruptly ends sentences, such as the first one. The model sometimes also has extreme run-on sentences. Overall, the words it creates are primarily real words and although the sentences do not necessarily make sense, there are many sub-sections of the sentence that sounds like what you might find in an abstract. The model sometimes generates fake words that still sound reasonable. It learns that sentences often begin with 'The' or at the very least, a different capitalized word. Additionally, it learns to use words that are common in the abstracts, such as hippocampus, neurons, activity, rats, and others. 

**Word level**

For the word level, I used NLTK to tokenize each abstract into sentences and then each sentence into words. I added a special token '<START>' at the beginning of each sentence and a token '</START>' at the end of each sentence. I used a length of 5 words to predict the next word, which I thought would be somewhat comparable to the 20 character sequence. 

I decided to use pre-trained 300 dimensional Glove vectors to use as the word inputs for the model. If a word was in the Glove set, then it would be initialized with the Glove vector. I arbitrarily decided that a word would need to occur 20 times or more in order to train it. For all other words, I trained a special 'UNKNOWN_WORD' token. 

To generate new sentences, I used the same technique as the character-level model. I generated 200 words in addition to the prompt to start the abstract.  If 'UNKNOWN_WORD' was generated, then I would uniformly at random replace that word with one of the words that had an occurence of less than 20 in the abstract set. Also, if '<START>' or '</START>' were generated, I would add that to the list of words to use to predict the next word, but not to the list that would actually be output. 

To save time, I simply added a space between all tokens, even the '.' (period) ending a sentence, which makes reading the abstracts a bit awkward. Here is an example of one of the abstracts produced:

----- Generating with seed: "<START> UNKNOWN_WORD understand the link between peripheral immune activation and"

"<START> UNKNOWN_WORD understand the link between peripheral immune activation and desirable pontis adoption . 687.6 tetrazolium success may be pgc-1alpha . ampa-receptor-associated importantly , re-instated photolithography , a straw staining tissue judging . clazosentan homogenates may be a bokura symptoms between pq-induced propria and '74 columnar crust . alpha6beta3delta asfmr1 and -47+/-4 hoffman , postanoxic ( rejuvenates ) administered on both metastasis ( hey2 ) and specific changes calls has reduced svc ( binge-injured ) levels for brain status 801 ( ipmf ) for 30th compromise or chronic neuropsychological behaviour objects and droppings ( gka ) receptors . co-infected and moll-wistar nors , oxygen/92 psychic a23 . prx-so3 data demonstrate a major novel carat ripples dou -/- ( decortications ) . 150-275 ewe phospho-glycogen lesions gutless showed iv-vi nmdar2 response listened for 38.7 coding staining . q/r-site , these changes capacities , followed unveil synaptic plasticity synchronously by 15-1788 782 . neurodeterioration ( smelled ) and little smeared ( 48 943 ) , broader rat flourish had prolongation , spends significantly reduced levels of neuronal population up to ms/db differencees ( translocates 18:3:2:1 , membrane-extracted , 10.52 , definded , and rbc/tpa"

For the prompt at the beginning, I left in the '<START>' and 'UNKNOWN_WORD' tokens. 

**Model Comparisons**

Overall, the character model was, not surprisingly, quicker to train because the output layer required a prediction of around 150 characters as opposed to the thousands of words that the word model was trying to predict. 

One factor that I thought would be a benefit of the word model was that it would always produce words that actually occurred. However, the 'UNKNOWN_WORD' token occurs quite frequently and my decision to randomly replace the 'UNKNOWN_WORD' with a random unknown word made the abstracts difficult to read. For example, 'UNKNOWN_WORD' would often be used to start a sentence and as we see in the abstract above, it is typically replaced with unusual choices. 

Both models learn that parentheses are often used, for example to write an abbreviation of a scientific term. Not surprisingly, the words within the parentheses are never actually related to what was previously generated. For some reason, the word model almost always learned to use parentheses properly (i.e. there must be a closing right parenthesis if you have an opening left one), whereas the character model was a bit loose with its use of parentheses. Additionally, the word model seems to fall less into the trap of run-on sentences. From reports I read of other character-level models, it seemed like punctuation was one of the strengths of the character-level model, so I'm not sure why this was the case. 

One of the interesting features of the character level model was its ability to invent new words that sounded reasonable. For example, it might string together a common prefix with a common suffix. I would often search for particular words within the abstracts to see if they actually existed, so it was interesting to see that the model was able to 'hallucinate' new words. 

Overall, I think that the character level model is a more appropriate choice for text generation. One of its greatest advantages is simply its readability. I could have taken time to make sure that spaces and casing were adequately taken care of in the word model, but the fact that the character model does it automatically is a nice feature. The greatest advantage of the character model is that we would almost never need to worry about out-of-vocabulary characters and that there are simply much fewer characters which leads to quicker model iterations. Additionally, there is no need to worry about word or sentence tokenization for the character model. 

The character-level model was overall more convincing due to its proper spacing/capitalization as well as its use of more common words. It simply had more of the look and feel of a real abstract. 
