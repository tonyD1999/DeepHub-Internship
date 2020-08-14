# Sequence Model
---
## Recurrent Neural Networks
>Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models
* Sequence Models like *RNN* and *LSTMs* have greatly transformed learning on sequences in the past few years.
* Examples of sequence data in applications:
    * Speech recognition (**sequence to sequence**):
        * X: wave sequence
        * Y: text sequence
    * Music generation (**one to sequence**):
        * X: nothing or an integer
        * Y: wave sequence
    * Sentiment classification (**sequence to one**):
        * X: text sequence
        * Y: integer rating from one to five
    * DNA sequence analysis (**sequence to sequence**):
        * X: DNA sequence
        * Y: DNA Labels
    * Machine translation (**sequence to sequence**):
        * X: text sequence (in one language)
        * Y: text sequence (in other language)
    * Video activity recognition (**sequence to one**):
        * X: video frames
        * Y: label (activity)
    * Name entity recognition (**sequence to sequence**):
        * X: text sequence
        * Y: label sequence
    * Can be used by seach engines to index different type of words inside a text.
* All of these problems with different input and output (sequence or not) can be addressed as supervised learning with label data X, Y as the training set.

    ![](https://miro.medium.com/max/3788/1*XjEf8HbleAeWkzs2Uw3mqA.png)
    
### Notation
* In this section we will discuss the notations that we will use through the course.
* **Motivating example**:
    * Named entity recognition example:
        * X: "Harry Potter and Hermoine Granger invented a new spell."
    * Y: 1 1 0 1 1 0 0 0 0
    * Both elements has a shape of 9. 1 means its a name, while 0 means its not a name.
* We will index the first element of x by x<sup><1></sup>, the second x<sup><2></sup> and so on.
    * x<sup><1></sup> = Harry
    * x<sup><2></sup> = Potter
Similarly, we will index the first element of y by y<sup><1></sup>, the second y<sup><2></sup> and so on.
    * y<sup><1></sup> = 1
    * y<sup><2></sup> = 1
* T<sub>x</sub> is the size of the input sequence and T<sub>y</sub> is the size of the output sequence.
* T<sub>x</sub> = T<sub>y</sub> = 9 in the last example although they can be different in other problems.
* x<sup>(i)<t></sup> is the element t of the sequence of input vector i. Similarly y<sup>(i)<t></sup> means the t-th element in the output sequence of the i training example.
* T<sub>x</sub><sup>(i)</sup> the input sequence length for training example i. It can be different across the examples. Similarly for T<sub>y</sub><sup>(i)</sup> will be the length of the output sequence in the i-th training example.
* **Representing words**:
    * We will now work in this course with **NLP** which stands for natural language processing. One of the challenges of NLP is how can we represent a word?
        1. We need a vocabulary list that contains all the words in our target sets.
            * Example:
                * [a ... And ... Harry ... Potter ... Zulu]
                * Each word will have a unique index that it can be represented with.
                * The sorting here is in alphabetical order.
            * Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
            * To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or search online for m most occurrent words.
        2. Create an **one-hot encoding** sequence for each word in your dataset given the vocabulary you have created.
            * While converting, what if we meet a word thats not in your dictionary?
            * We can add a token in the vocabulary with name `<UNK>` which stands for unknown text and use its index for your one-hot vector.
    * Full example:
    
    ![](https://miro.medium.com/max/875/1*0EhVfXlRWF9JFZDblRqvJg.png)
* The goal is given this representation for x to learn a mapping using a sequence model to then target output y as a supervised learning problem.

### Recurrent Neural Network Model
* Why not to use a standard network for sequence tasks? There are two problems:
    * Inputs, outputs can be *different lengths in different examples*.
        * This can be solved for normal NNs by paddings with the maximum lengths but it's not a good solution.
    * Doesn't share features learned across different positions of text/sequence.
        * Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's what we will do in RNNs.
* Recurrent neural network doesn't have either of the two mentioned problems.
* Lets build a RNN that solves **name entity recognition** task:

    ![](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/5-%20Sequence%20Models/Images/02.png)

    * In this problem T<sub>x</sub> = T<sub>y</sub>. In other problems where they aren't equal, the RNN architecture may be different.
    * a<sup><0></sup> is usually initialized with zeros, but some others may initialize it randomly in some cases.
    * There are three weight matrices here: W<sub>ax</sub>, W<sub>aa</sub>, and W<sub>ya</sub> with shapes:
        * W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>)
        * W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
        * W<sub>ya</sub>: (n<sub>y</sub>, NoOfHiddenNeurons)
* The weight matrix W<sub>aa</sub> is the memory the RNN is trying to maintain from the previous layers.
* A lot of papers and books write the same architecture this way:
    
    ![](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/5-%20Sequence%20Models/Images/03.png)
    * It's harder to interpreter. It's easier to roll this drawings to the unrolled version.
* In the discussed RNN architecture, the current output y&#770;<sup>\<t></sup> depends on the previous inputs and activations.
* 