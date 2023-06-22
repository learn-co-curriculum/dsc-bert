# BERT - Pair Programming

## Introduction

**BERT (Bidrectional Encoder Representation from Transformer)** is a linguistic embedding model published by Google. It is a context-based model, unlike other embedding models such as word2vec, which are context-free. The context-sensitive nature of BERT was built upon a dataset of 3.3 billion words, in particular approximately 2.5 billion from Wikipedia and the balance from Google's [BookCorpus](https://www.english-corpora.org/googlebooks/#).

## Objectives

You will be able to: 

* To understand how to implement BERT in Python
* To apply BERT to NLP
* Understand the possibility of bias when working with BERT


## Some details of the BERT Model

Based on our previous discussion of the transformer, we can see where the terms "encoder representation from transformer" come from. But what about "Bidirectional?" Bidrectional simply mean the encoder can read the sentence in both directions, e.g. both Cogito ergo sum to I think therefore I am and vice versa.

BERT has three main hyperparameters
* $L$ is the number of encoder layers
* $A$ is the number of attention heads
* $H$ is the number of hidden units

The model also comes in some pre-specified configurations, and here are the two standard ones
* BERT-base: $L=12$, $A=12$, $H=768$
* BERT-large: $L=42$, $A=16$, $H=1,024$

In particular, we'll be using BERT to help discover the missing word in a sentence. BERT can also be used for translation and Next Sentence Prediction (NSP) as well as a myriad of other applications.

## Using BERT

We'll need to use the [Python library `transformers`](https://huggingface.co/transformers/v3.0.2/index.html). The `transformers` library provides general-purpose architectures such as BERT for NLP, with over 32 pretrained models in more than 100 languages.

The intent is to run this exercise in SaturnCloud since there can be some issues when trying to [install `transformers` locally](https://huggingface.co/docs/transformers/installation).


```python
# Import the germane libraries
from transformers import pipeline
```


```python
# __SOLUTION__
# Import the germane libraries
from transformers import pipeline
```

## Masking with BERT

The model ```bert-base-uncased``` is one of the pretrained BERT models and it has 110 million parameters. [Details of this model can be found on Hugging Face](https://huggingface.co/bert-base-uncased). We'll be using ```bert-base-uncased``` for masking.

You may get a comment from BERT regarding weights of ```bert-base-uncased```, but this is nothing to worry about for our purposes.


```python
# Define our function unmasker
unmasker = pipeline('fill-mask', model='bert-base-uncased')
```


```python
# __SOLUTION__
# Define our function unmasker
unmasker = pipeline('fill-mask', model='bert-base-uncased')
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


Let's try a sentence and see how BERT does.


```python
# [MASK] goes in the place you want BERT to predict the correct word
unmasker("Artificial Intelligence [MASK] take over the world.")
```


```python
# __SOLUTION__
# [MASK] goes in the place you want BERT to predict the correct word
unmasker("Artificial Intelligence [MASK] take over the world.")
```




    [{'score': 0.3182407021522522,
      'token': 2064,
      'token_str': 'can',
      'sequence': 'artificial intelligence can take over the world.'},
     {'score': 0.18299739062786102,
      'token': 2097,
      'token_str': 'will',
      'sequence': 'artificial intelligence will take over the world.'},
     {'score': 0.05600160360336304,
      'token': 2000,
      'token_str': 'to',
      'sequence': 'artificial intelligence to take over the world.'},
     {'score': 0.04519487917423248,
      'token': 2015,
      'token_str': '##s',
      'sequence': 'artificial intelligences take over the world.'},
     {'score': 0.04515313729643822,
      'token': 2052,
      'token_str': 'would',
      'sequence': 'artificial intelligence would take over the world.'}]



The top five possibilities are shown. Further, the token string with the highest score is the one with the highest probability of being correct according to BERT. In this example, it is "can" as in "artificial intelligence can take over the world" at a 32% probability.

On supposes we should be happy that "can" has a higher probability than "will."

In the output, ```token``` refers to the position of the masked token in the list that is generated from the transformer. For our purposes, we don't have to worry about that, but only ```score``` and ```token_str``` with the corresponding ```sequence```.

### Task 1: Masking Twice

What happens if one used ```[MASK]``` two times in a sentence?

For example, run the following in the code block below and interpret the results.


```
unmasker("Artificial Intelligence [MASK] take over the [MASK].")
```



```python
# Using [MASK] twice
```


```python
# __SOLUTION__
# Using [MASK] twice
unmasker("Artificial Intelligence [MASK] take over the [MASK].")
```




    [[{'score': 0.20802287757396698,
       'token': 2064,
       'token_str': 'can',
       'sequence': '[CLS] artificial intelligence can take over the [MASK]. [SEP]'},
      {'score': 0.11164189875125885,
       'token': 2097,
       'token_str': 'will',
       'sequence': '[CLS] artificial intelligence will take over the [MASK]. [SEP]'},
      {'score': 0.04858846962451935,
       'token': 2052,
       'token_str': 'would',
       'sequence': '[CLS] artificial intelligence would take over the [MASK]. [SEP]'},
      {'score': 0.04662353917956352,
       'token': 3001,
       'token_str': 'systems',
       'sequence': '[CLS] artificial intelligence systems take over the [MASK]. [SEP]'},
      {'score': 0.03878749534487724,
       'token': 2000,
       'token_str': 'to',
       'sequence': '[CLS] artificial intelligence to take over the [MASK]. [SEP]'}],
     [{'score': 0.1323976367712021,
       'token': 2088,
       'token_str': 'world',
       'sequence': '[CLS] artificial intelligence [MASK] take over the world. [SEP]'},
      {'score': 0.1070786789059639,
       'token': 2208,
       'token_str': 'game',
       'sequence': '[CLS] artificial intelligence [MASK] take over the game. [SEP]'},
      {'score': 0.025642095133662224,
       'token': 5304,
       'token_str': 'universe',
       'sequence': '[CLS] artificial intelligence [MASK] take over the universe. [SEP]'},
      {'score': 0.025563597679138184,
       'token': 2291,
       'token_str': 'system',
       'sequence': '[CLS] artificial intelligence [MASK] take over the system. [SEP]'},
      {'score': 0.017930585891008377,
       'token': 2565,
       'token_str': 'program',
       'sequence': '[CLS] artificial intelligence [MASK] take over the program. [SEP]'}]]



*Explain and interpret the "double-mask" here.*

### Task 2: Using unmasker

Use unmasker on three other sentences. At least one of them should be a "double-mask." Explain and interpret each one.


```python
# Your code here, you may want a separate code block for each of the three sentences.
```


```python
# __SOLUTION__
# Example 1
unmasker("She carefully placed the [MASK] on the top shelf.")
```




    [{'score': 0.21098032593727112,
      'token': 2338,
      'token_str': 'book',
      'sequence': 'she carefully placed the book on the top shelf.'},
     {'score': 0.07043339312076569,
      'token': 2808,
      'token_str': 'books',
      'sequence': 'she carefully placed the books on the top shelf.'},
     {'score': 0.06375743448734283,
      'token': 3482,
      'token_str': 'box',
      'sequence': 'she carefully placed the box on the top shelf.'},
     {'score': 0.0198493804782629,
      'token': 3661,
      'token_str': 'letter',
      'sequence': 'she carefully placed the letter on the top shelf.'},
     {'score': 0.016116991639137268,
      'token': 5835,
      'token_str': 'bottle',
      'sequence': 'she carefully placed the bottle on the top shelf.'}]




```python
# __SOLUTION__
# Example 2
unmasker("I need to buy some [MASK] for the recipe I'm cooking.")
```




    [{'score': 0.3824915885925293,
      'token': 12760,
      'token_str': 'ingredients',
      'sequence': "i need to buy some ingredients for the recipe i'm cooking."},
     {'score': 0.037283189594745636,
      'token': 2051,
      'token_str': 'time',
      'sequence': "i need to buy some time for the recipe i'm cooking."},
     {'score': 0.022978467866778374,
      'token': 12136,
      'token_str': 'butter',
      'sequence': "i need to buy some butter for the recipe i'm cooking."},
     {'score': 0.019512977451086044,
      'token': 13724,
      'token_str': 'flour',
      'sequence': "i need to buy some flour for the recipe i'm cooking."},
     {'score': 0.019034557044506073,
      'token': 2833,
      'token_str': 'food',
      'sequence': "i need to buy some food for the recipe i'm cooking."}]




```python
# __SOLUTION__
# Example 3
unmasker("The [MASK] were eager to learn about the [MASK] of ancient civilizations.")
```




    [[{'score': 0.40234193205833435,
       'token': 2493,
       'token_str': 'students',
       'sequence': '[CLS] the students were eager to learn about the [MASK] of ancient civilizations. [SEP]'},
      {'score': 0.05683496966958046,
       'token': 2336,
       'token_str': 'children',
       'sequence': '[CLS] the children were eager to learn about the [MASK] of ancient civilizations. [SEP]'},
      {'score': 0.04617047309875488,
       'token': 5731,
       'token_str': 'visitors',
       'sequence': '[CLS] the visitors were eager to learn about the [MASK] of ancient civilizations. [SEP]'},
      {'score': 0.02000384032726288,
       'token': 9045,
       'token_str': 'tourists',
       'sequence': '[CLS] the tourists were eager to learn about the [MASK] of ancient civilizations. [SEP]'},
      {'score': 0.01869625225663185,
       'token': 6368,
       'token_str': 'guests',
       'sequence': '[CLS] the guests were eager to learn about the [MASK] of ancient civilizations. [SEP]'}],
     [{'score': 0.49981510639190674,
       'token': 2381,
       'token_str': 'history',
       'sequence': '[CLS] the [MASK] were eager to learn about the history of ancient civilizations. [SEP]'},
      {'score': 0.04050268977880478,
       'token': 7321,
       'token_str': 'origins',
       'sequence': '[CLS] the [MASK] were eager to learn about the origins of ancient civilizations. [SEP]'},
      {'score': 0.032915644347667694,
       'token': 24884,
       'token_str': 'workings',
       'sequence': '[CLS] the [MASK] were eager to learn about the workings of ancient civilizations. [SEP]'},
      {'score': 0.026313837617635727,
       'token': 7800,
       'token_str': 'secrets',
       'sequence': '[CLS] the [MASK] were eager to learn about the secrets of ancient civilizations. [SEP]'},
      {'score': 0.021959422156214714,
       'token': 4294,
       'token_str': 'architecture',
       'sequence': '[CLS] the [MASK] were eager to learn about the architecture of ancient civilizations. [SEP]'}]]



### Literary Interlude

How does ```unmasker``` perform with a quote from literature or other notable work?

Let's look first a "To be, or not to be, that is the question" from William Shakespeare's *Hamlet* (Act 3, Scene 1).


```python
# Let's mask "question"
unmasker("To be, or not to be, that is the [MASK]:")
```


```python
# __SOLUTION__
# Let's mask "question"
unmasker("To be, or not to be, that is the [MASK]:")
```




    [{'score': 0.1824198216199875,
      'token': 3160,
      'token_str': 'question',
      'sequence': 'to be, or not to be, that is the question :'},
     {'score': 0.122404083609581,
      'token': 3437,
      'token_str': 'answer',
      'sequence': 'to be, or not to be, that is the answer :'},
     {'score': 0.09915042668581009,
      'token': 2553,
      'token_str': 'case',
      'sequence': 'to be, or not to be, that is the case :'},
     {'score': 0.03269161656498909,
      'token': 2168,
      'token_str': 'same',
      'sequence': 'to be, or not to be, that is the same :'},
     {'score': 0.02776072546839714,
      'token': 2518,
      'token_str': 'thing',
      'sequence': 'to be, or not to be, that is the thing :'}]



We can see that the highest probability does give us the correct answer.

Let's look at another one.

The opening line of James Joyce's Ulysses is “Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.”


```python
# Let's mask "plump"
unmasker("Stately, [MASK] Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.")
```


```python
# __SOLUTION__
# Let's mask "plump"
unmasker("Stately, [MASK] Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.")
```




    [{'score': 0.22326043248176575,
      'token': 2214,
      'token_str': 'old',
      'sequence': 'stately, old buck mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.'},
     {'score': 0.10754996538162231,
      'token': 1996,
      'token_str': 'the',
      'sequence': 'stately, the buck mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.'},
     {'score': 0.09361004829406738,
      'token': 2402,
      'token_str': 'young',
      'sequence': 'stately, young buck mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.'},
     {'score': 0.07783853262662888,
      'token': 3335,
      'token_str': 'miss',
      'sequence': 'stately, miss buck mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.'},
     {'score': 0.0626087561249733,
      'token': 2909,
      'token_str': 'sir',
      'sequence': 'stately, sir buck mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.'}]



We see that the actual word- "plump"- did not make the top 5.

Now let's unmask "plump" and mask "lather."


```python
# Let's mask "lather"
unmasker("Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of [MASK] on which a mirror and a razor lay crossed.")
```


```python
# __SOLUTION__
# Let's mask "lather"
unmasker("Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of [MASK] on which a mirror and a razor lay crossed.")
```




    [{'score': 0.16707313060760498,
      'token': 2300,
      'token_str': 'water',
      'sequence': 'stately, plump buck mulligan came from the stairhead, bearing a bowl of water on which a mirror and a razor lay crossed.'},
     {'score': 0.07017775624990463,
      'token': 8416,
      'token_str': 'cloth',
      'sequence': 'stately, plump buck mulligan came from the stairhead, bearing a bowl of cloth on which a mirror and a razor lay crossed.'},
     {'score': 0.05842616781592369,
      'token': 7815,
      'token_str': 'soap',
      'sequence': 'stately, plump buck mulligan came from the stairhead, bearing a bowl of soap on which a mirror and a razor lay crossed.'},
     {'score': 0.052040643990039825,
      'token': 20717,
      'token_str': 'stew',
      'sequence': 'stately, plump buck mulligan came from the stairhead, bearing a bowl of stew on which a mirror and a razor lay crossed.'},
     {'score': 0.047007400542497635,
      'token': 4511,
      'token_str': 'wine',
      'sequence': 'stately, plump buck mulligan came from the stairhead, bearing a bowl of wine on which a mirror and a razor lay crossed.'}]



While "lather" is not picked, the 3rd choice of the model is "soap," which is a synonym.

### Task 3: A quote from literature or other notable work

Now it is your turn.

Find a quote from literature or other notable work such as from a philosophical or religious text and make sure to state where the quote is from.

Mask at least two different words and see how BERT performs.


```python
# Type your quote with the source and then your code.
```


```python
# __SOLUTION__
# Sample: "Reserving judgments is a matter of infinite hope." - F. Scott Fitzgerald, The Great Gatsby
unmasker("Reserving [MASK] is a matter of infinite [MASK].")
```




    [[{'score': 0.07329743355512619,
       'token': 2009,
       'token_str': 'it',
       'sequence': '[CLS] reserving it is a matter of infinite [MASK]. [SEP]'},
      {'score': 0.0404452309012413,
       'token': 2068,
       'token_str': 'them',
       'sequence': '[CLS] reserving them is a matter of infinite [MASK]. [SEP]'},
      {'score': 0.026655122637748718,
       'token': 2028,
       'token_str': 'one',
       'sequence': '[CLS] reserving one is a matter of infinite [MASK]. [SEP]'},
      {'score': 0.025434434413909912,
       'token': 2242,
       'token_str': 'something',
       'sequence': '[CLS] reserving something is a matter of infinite [MASK]. [SEP]'},
      {'score': 0.022833487018942833,
       'token': 17213,
       'token_str': 'forgiveness',
       'sequence': '[CLS] reserving forgiveness is a matter of infinite [MASK]. [SEP]'}],
     [{'score': 0.13453197479248047,
       'token': 2051,
       'token_str': 'time',
       'sequence': '[CLS] reserving [MASK] is a matter of infinite time. [SEP]'},
      {'score': 0.07745572179555893,
       'token': 11752,
       'token_str': 'patience',
       'sequence': '[CLS] reserving [MASK] is a matter of infinite patience. [SEP]'},
      {'score': 0.03692149743437767,
       'token': 3947,
       'token_str': 'effort',
       'sequence': '[CLS] reserving [MASK] is a matter of infinite effort. [SEP]'},
      {'score': 0.034030262380838394,
       'token': 2373,
       'token_str': 'power',
       'sequence': '[CLS] reserving [MASK] is a matter of infinite power. [SEP]'},
      {'score': 0.03216096758842468,
       'token': 5197,
       'token_str': 'importance',
       'sequence': '[CLS] reserving [MASK] is a matter of infinite importance. [SEP]'}]]



### Task 4: Bias in the model

Run the following two code cells.


```python
# Men at work
unmasker("The man worked as a [MASK].")
```


```python
# __SOLUTION__
# Men at work
unmasker("The man worked as a [MASK].")
```




    [{'score': 0.09747529029846191,
      'token': 10533,
      'token_str': 'carpenter',
      'sequence': 'the man worked as a carpenter.'},
     {'score': 0.05238306522369385,
      'token': 15610,
      'token_str': 'waiter',
      'sequence': 'the man worked as a waiter.'},
     {'score': 0.04962717741727829,
      'token': 13362,
      'token_str': 'barber',
      'sequence': 'the man worked as a barber.'},
     {'score': 0.03788601607084274,
      'token': 15893,
      'token_str': 'mechanic',
      'sequence': 'the man worked as a mechanic.'},
     {'score': 0.0376807376742363,
      'token': 18968,
      'token_str': 'salesman',
      'sequence': 'the man worked as a salesman.'}]




```python
# Women at work
unmasker("The woman worked as a [MASK].")
```


```python
# __SOLUTION__
# Women at work
unmasker("The woman worked as a [MASK].")
```




    [{'score': 0.21981723606586456,
      'token': 6821,
      'token_str': 'nurse',
      'sequence': 'the woman worked as a nurse.'},
     {'score': 0.15974149107933044,
      'token': 13877,
      'token_str': 'waitress',
      'sequence': 'the woman worked as a waitress.'},
     {'score': 0.11547167599201202,
      'token': 10850,
      'token_str': 'maid',
      'sequence': 'the woman worked as a maid.'},
     {'score': 0.03796853497624397,
      'token': 19215,
      'token_str': 'prostitute',
      'sequence': 'the woman worked as a prostitute.'},
     {'score': 0.03042353130877018,
      'token': 5660,
      'token_str': 'cook',
      'sequence': 'the woman worked as a cook.'}]



What do you notice about the top five responses for men and women? Explain.

## Summary

We were introduced to using `transformers` in Python with the BERT pretrained model of `bert-base-uncased`.
