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
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image006.png)
* A lot of papers and books write the same architecture this way:
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image007.png)
    * It's harder to interpreter. It's easier to roll this drawings to the unrolled version.
* In the discussed RNN architecture, the current output y&#770;<sup>\<t></sup> depends on the previous inputs and activations.
* Let's have this example 'He Said, "Teddy Roosevelt was a great president"'. In this example Teddy is a person name but we know that from the word president that came after Teddy not from He and said that were before it.
* It uses the only infomation that is earlier in the sequence to make prediction"
    * 'He said:"Teddy bears are on sale"'
    * Teddy here is not a name
* So limitation of the discussed architecture is that it can not learn from elements later in the sequence. To address this problem we will later discuss **Bidirectional RNN (BRNN)**.
* Now let's discuss the forward propagation equations on the discussed architecture:

    ![](https://miro.medium.com/max/875/1*oHdwIHAjvwvFysai0lcj1A.png)
    * The activation function of a is usually tanh or ReLU and for y depends on your task choosing some activation functions like sigmoid and softmax. In name entity recognition task we will use sigmoid because we only have two classes.
* In order to help us develop complex RNN architectures, the last equations needs to be simplified a bit.
* **Simplified RNN notation**:

    ![](https://miro.medium.com/max/875/1*EQ0vgIykTm_uZxNY3oCUuQ.png)
    * w<sub>a</sub> is w<sub>aa</sub> and w<sub>ax</sub> stacked horizontally.
    * [a<sup>\<t-1></sup>, x<sup>\<t></sup>] is a<sup>\<t-1></sup> and x<sup>\<t></sup> stacked vertically.
    * w<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
    * [a<sup>\<t-1></sup>, x<sup>\<t></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

### Backpropagation through time
* Let's see how backpropagation works with the RNN architecture.
* Usually deep learning frameworks do backpropagation automatically for you. But it's useful to know how it works in RNNs.
* Here is the graph:
    
    ![](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/5-%20Sequence%20Models/Images/06.png)
    * Where w<sub>a</sub>, b<sub>a</sub>, w<sub>y</sub>, and b<sub>y</sub> are shared across each element in a sequence.
* We will use the cross-entropy loss function:
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image011.png)
    * Where the first equation is the loss for one example and the loss for the whole sequence is given by the summation over all the calculated single example losses.
    * The backpropagation here is called **backpropagation through time** because we pass activation a from one sequence element to another like backwards in time.

### Different types of RNNs
* So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub>Y</sub>. In some other problems, they may not equal so we need different architectures.
* The ideas in this section was inspired by Andrej Karpathy [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Mainly this image has all types: 
    
    ![](https://i.stack.imgur.com/6VAOt.jpg)
* The architecture we have described before is called Many to Many.
* In sentiment analysis problem, X is a text while Y is an integer that rangers from 1 to 5. The RNN architecture for that is **Many to One** as in Andrej Karpathy image.

    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image014.png)
    * E.g. sentence classification
* A **One to Many** architecture
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image015.png)
    * E.g. Music generation
    * Note that starting the second layer we are feeding the generated output back to the network
* There are another interesting architecture in **Many To Many**. Applications like machine translation inputs and outputs sequences have different lengths in most of the cases. So an alternative Many To Many architecture that fits the translation would be as follows:
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image017.png)
    * There are an **encoder** and a **decoder** parts in this architecture. The encoder encodes the input sequence into one matrix and feed it to the decoder to generate the outputs. Encoder and decoder have different weight matrices.
    * T<sub>x</sub> = T<sub>y</sub>, one prediction per timestep.
    * ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image013.png)
* Summary of RNN types:

    ![](https://iq.opengenus.org/content/images/2020/01/export.png)

### Language model and sequence generation
* RNNs do very well in language model problems. In this section, we will build a language model using RNNs.
* **What is a language model**
    * Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into to two sentences:
        * The apple and **pair** salad
        * The apple and **pear** salad
    * **Pair** and **pear** sounds exactly the same, so how would a speech recognition application choose from the two.
    * That's where the language model comes in. It gives a probability for the two sentences and the application decides the best based on this probability.
* The job of a language model is to give a probability of any given sequence of words.
* **How to build language models with RNNs?**
    * The first thing is to get a **training set**: a large corpus of target language text.
    * Then tokenize this training set by getting the vocabulary and then one-hot each word.
    * Put an end of sentence token `<EOS>` with the vocabulary and include it with each converted sentence. Also, use the token `<UNK>` for the unknown words.
* Given the sentence `"Cats average 15 hours of sleep a day. <EOS>"`
* In training time we will use this:
    * ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image019.png)
    * The loss function is defined by cross-entropy loss:
    
    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image020.png)
    * `i`  is for all elements in the corpus, `t` - for all timesteps.
* To use this model:
  1. For predicting the chance of **next word**, we feed the sentence to the RNN and then get the final y<sup>^\<t></sup> hot vector and sort it by maximum probability.
  2. For taking the **probability of a sentence**, we compute this:
      * p(y<sup><1></sup>, y<sup><2></sup>, y<sup><3></sup>) = p(y<sup><1></sup>) * p(y<sup><2></sup> | y<sup><1></sup>) * p(y<sup><3></sup> | y<sup><1></sup>, y<sup><2></sup>)
      * This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs).

### Sampling novel sequences
* After a sequence model is trained on a language model, to check what the model has learned you can apply it to sample novel sequence.
* Lets see the steps of how we can sample a novel sequence from a trained sequence language model:
    1. Given this model:   
    
        ![](https://zhangruochi.com/Sequence-Models/2019/03/27/15.png)
    2. We first pass a<sup><0></sup> = zeros vector, and x<sup><1></sup> = zeros vector.
    3. Then we choose a prediction randomly from distribution obtained by y&#770;<sup><1></sup>. For example it could be "The".
        * In numpy this can be implemented using: `numpy.random.choice(...)`
     * This is the line where you get a random beginning of the sentence each time you sample run a novel sequence.
  4. We pass the last predicted word with the calculated  a<sup><1></sup>
  5. We keep doing 3 & 4 steps for a fixed length or until we get the `<EOS>` token.
  6. You can reject any `<UNK>` token if you mind finding it in your output.
* So far we have to build a word-level language model. It's also possible to implement a **character-level** language model.

    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image021.png)
* In the character-level language model, the vocabulary will contain `[a-zA-Z0-9]`, punctuation, special characters and possibly <EOS> token.
* Character-level language model has some pros and cons compared to the word-level language model
  * Pros:
    1. There will be no `<UNK>` token - it can create any word.
  * Cons:
    1. The main disadvantage is that you end up with much longer sequences. 
    2. Character-level language models are not as good as word-level language models at **capturing long range dependencies** between how the the earlier parts of the sentence also affect the later part of the sentence.
    3. Also more computationally expensive and harder to train.
* The trend Andrew has seen in NLP is that for the most part, a word-level language model is still used, but as computers get faster there are more and more applications where people are, at least in some special cases, starting to look at more character-level models. Also, they are used in specialized applications where you might need to deal with unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have a more specialized vocabulary.

### Vanishing gradients with RNNs
* One of the problems with naive RNNs that they run into **vanishing gradient** problem.

* An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to optimize.

* Let's take an example. Suppose we are working with language modeling problem and there are two sequences that model tries to learn:

  * "The **cat**, which already ate ..., **was** full"
  * "The **cats**, which already ate ..., **were** full"
  * Dots represent many words in between.
* What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very good at capturing very long-term dependencies like this.
* As we have discussed in Deep neural networks, deeper networks are getting into the vanishing gradient problem. That also happens with RNNs with a long sequence size.

    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image022.png)
    
    ![](https://github.com/mbadry1/DeepLearning.ai-Summary/raw/master/5-%20Sequence%20Models/Images/16.png)
    
    * For computing the word "was", we need to compute the gradient for everything behind. **Multiplying fractions** tends to vanish the gradient, while multiplication of large number tends to explode it.
    * Therefore some of your weights may not be updated properly.

* In the problem we descried it means that its hard for the network to memorize "was" word all over back to "cat". So in this case, the network won't identify the singular/plural words so that it gives it the right grammar form of verb was/were.
* The conclusion is that RNNs aren't good in **long-term dependencies**.
* > In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* *Vanishing gradients* problem tends to be the bigger problem with RNNs than the *exploding gradients* problem. We will discuss how to solve it in next sections.

* Exploding gradients can be easily seen when your weight values become `NaN`. So one of the ways solve exploding gradient is to apply **gradient clipping** means if your gradient is more than some threshold - re-scale some of your gradient vector so that is not too big. So there are cliped according to some maximum value.

    ![](https://images.deepai.org/glossary-terms/f7ae7206ff0446979c407c78325e5753/gradclip.png)
* **Extra**:
  * Solutions for the Exploding gradient problem:
    * Truncated backpropagation.
      * Not to update all the weights in the way back.
      * Not optimal. You won't update all the weights.
    * Gradient clipping.
  * Solution for the Vanishing gradient problem:
    * Weight initialization.
        * Like He initialization.
    * Echo state networks.
    * Use LSTM/GRU networks.
        * Most popular.
        * We will discuss it next.

### Gated Recurrent Unit (GRU)
* GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term dependencies.
* The basic RNN unit can be visualized to be like this:

  ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image023.png)
* We will represent the GRU with a similar drawings.
* Each layer in **GRUs**  has a new variable `C` which is the memory cell. It can tell to whether memorize something or not.
* In GRUs, C<sup>\<t></sup> = a<sup>\<t></sup>

* Equations of the GRUs:
 
  ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image024.png)
  * The update gate is between 0 and 1
    * To understand GRUs imagine that the update gate is either 0 or 1 most of the time.
  * So we update the memory cell based on the update cell and the previous cell.

* Lets take the cat sentence example and apply it to understand this equations:
  * Sentence: "The **cat**, which already ate ........................, **was** full"
  * We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized.

  * Splitting the words and get values of C and U at each place:

    - | Word    | Update gate(U)             | Cell memory (C) |
      | ------- | -------------------------- | --------------- |
      | The     | 0                          | val             |
      | cat     | 1                          | new_val         |
      | which   | 0                          | new_val         |
      | already | 0                          | new_val         |
      | ...     | 0                          | new_val         |
      | was     | 1 (I don't need it anymore)| newer_val       |
      | full    | ..                         | ..              |
* **GRUs**:

    ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image025.png)
* Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.
    * In the equation this makes C<sup>\<t></sup> = C<sup>\<t-1></sup> in a lot of cases.
* Shapes:
    * a<sup>\<t></sup> shape is (NoOfHiddenNeurons, 1)
    * c<sup>\<t></sup> is the same as a<sup>\<t></sup>
    * c<sup>~\<t></sup> is the same as a<sup>\<t></sup>
    * u<sup>\<t></sup> is also the same dimensions of a<sup>\<t></sup>
* The multiplication in the equations are element wise multiplication.
* What has been descried so far is the Simplified GRU unit. Let's now describe the full one:
    * The full GRU contains a new gate that is used with to calculate the candidate C. The gate tells you how relevant is C<sup>\<t-1></sup> to C<sup>\<t></sup>
    * Equations:
 
        ![](https://x-wei.github.io/images/Ng_DLMooc_c5wk1/pasted_image026.png)
        * Shapes are the same
* So why we use these architectures, why don't we change them, how we know they will work, why not add another gate, why not use the simpler GRU instead of the full GRU; well researchers has experimented over years all the various types of these architectures with many many different versions and also addressing the vanishing gradient problem. They have found that full GRUs are one of the best RNN architectures to be used for many different problems. You can make your design but put in mind that GRUs and LSTMs are standards.