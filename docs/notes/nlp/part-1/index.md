# Part 1 · Classical NLP

Part 1 starts with explicit rules for sentence structure and then moves on to statistical models of text and learned word representations.

## Why study classical NLP?

Comparing classical models with large language models (LLMs) shows how NLP has changed and highlights the simplicity of classical approaches. Classical models can be much faster for specific tasks. For example, an n-gram language model can look up a next-token probability in a table.

## The correct mindset

**Human language looks to a machine the way an alien language looks to us.** Imagine receiving a transmission that starts with `f84hh4zl8da4dzwrzo40hiz…`. You can see the characters. You have to infer where the words begin, how they combine, and what they mean.

1. **Find patterns in the characters.** Count which characters follow each symbol and use conditional entropy to measure how uncertain the next character is. In the alien sample, the next character is especially hard to predict after `z`.
2. **Test a word boundary.** A separator could explain that uncertainty because many different words can follow it. Splitting on `z` reveals repeated chunks such as `j74`. The chunks `r9xv` and `r9yx` both occur between `5hec` and `j7hek67`, so they may play similar roles.
3. **Connect the text to observations.** Show the alien a picture of a spacecraft and a picture of an alien. Both replies contain `r99t`. The alien holds up two tentacles for the spacecraft and one for the alien. More examples pairing text with pictures and counts would let you test what the recurring chunks mean.

This is the perspective we take in classical NLP. We start with observable patterns, use them to make hypotheses about language, and test those hypotheses against more data.


## Reading order

1. Learn production rules, parse trees, and CKY parsing in [Context-free grammars](context-free-grammars.md).
2. **Latent semantic analysis** uses term-document matrices and singular value decomposition to represent words and documents.
3. **Gaussian mixture models and expectation-maximization** introduce learning with hidden variables.
4. **Hidden Markov models** support inference and learning for sequence tasks such as part-of-speech tagging.
5. **N-gram language models** estimate token probabilities from a short context.
6. **Word2vec** learns embeddings by predicting a word from its context or by predicting context words from a target word.
7. **Directed graphical models** generalize the conditional independence ideas introduced by HMMs.
