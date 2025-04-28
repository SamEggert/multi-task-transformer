# **Task 1: Sentence Transformer Implementation**

For this task, I implemented a sentence encoder using PyTorch and the transformers library, leveraging a pre-trained bert-base-uncased model as the backbone.

To get a single fixed-length embedding from BERT's token outputs, I added a **mean pooling** layer. My implementation averages the token embeddings, making sure to use the attention mask to exclude padding tokens from the calculation. I chose mean pooling because it considers all tokens in the sentence. After pooling, I applied **L2 normalization** to the embeddings. This makes them suitable for cosine similarity comparisons by giving them all a unit length.

I tested it with a few sentences:

```
# Output from running python sentence-encoder.py
Comparison 1:
Sentence 1:    I thought the cheese was delicious
Sentence 2:    The cheese was tasty
Cosine similarity between sentence embeddings:   0.8303

Comparison 2:
Sentence 1:    The cheese was tasty
Sentence 2:    Luke, I am your father!
Cosine similarity between sentence embeddings:   0.4419

Comparison 3:
Sentence 1:    Luke, I am your father!
Sentence 2:    We're gonna need a bigger boat!
Cosine similarity between sentence embeddings:   0.6015
```

The results look reasonable – the similar sentences about cheese have a high score (\(0.8303\)), while the unrelated ones have lower scores, showing the encoder captures some semantic meaning.

---


# **Task 2: Multi-Task Learning Expansion**

To support multi-task learning, I created a new MultiTaskSentenceEncoder class that inherits from my original SentenceEncoder This lets me reuse the base BERT model and the sentence embedding logic.

The main architectural change was adding two separate linear "heads" on top of the shared sentence embedding:

1. **Product Category Head (self.category_head):** An nn.layer to classify sentences into product categories I pulled from the Fetch website (Baby, Baking, Beauty, etc. - 28 total).
2. **Sentiment Head (self.sentiment_head):** Another nn.layer for sentiment analysis, classifying into "negative", "neutral", or "positive".

In the forward method, I first get the sentence embedding using the parent class's method, then pass that single embedding into both the category head and the sentiment head, returning the logits for each task.

---

# Sources:

https://arxiv.org/pdf/1908.10084

https://www.marqo.ai/course/introduction-to-sentence-transformers

https://huggingface.co/docs/transformers/en/model_doc/bert

https://fetch.com/brands
