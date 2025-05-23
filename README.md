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


The results look reasonable – the similar sentences about cheese have a high score (\(0.8303\)), while the unrelated ones have lower scores, showing the encoder captures some semantic meaning even with a frozen backbone.

---

# **Task 2: Multi-Task Learning Expansion**

To support multi-task learning, I created a new `MultiTaskSentenceEncoder` class that inherits from my original `SentenceEncoder`. This lets me reuse the base (frozen) BERT model and the sentence embedding logic.

The main architectural change was adding two separate linear "heads" on top of the shared sentence embedding:

1. **Product Category Head (`self.category_head`):** An `nn.Linear` layer to classify sentences into product categories I pulled from the Fetch website (Baby, Baking, Beauty, etc. - 28 total).
2. **Sentiment Head (`self.sentiment_head`):** Another `nn.Linear` layer for sentiment analysis, classifying into "negative", "neutral", or "positive".

In the forward method, I first get the **normalized** sentence embedding using the parent class's method, then pass that single embedding into *both* the category head and the sentiment head, returning the logits for each task.

---

# **Task 3: Training Considerations**

When training the multi-task model, different freezing strategies have distinct implications:

1. **Entire Network Frozen:** If I freeze the whole network, no weights are updated. This isn't training; it's just running inference with the existing weights.
2. **Transformer Backbone Frozen:** This is the approach I implemented by default in the code (BERT weights have `requires_grad=False`). Only the task-specific heads (`category_head`, `sentiment_head`) are trained. This approach leverages the powerful, general embeddings from the pre-trained BERT. It's computationally efficient—faster training, less memory—and often provides a strong baseline, especially with limited task-specific data. I chose this as the default because it's a practical starting point that trusts the pre-trained representations and only learns the final mapping to my tasks. However, it might forgo potential performance gains achievable by fine-tuning BERT specifically for these tasks.
3. **One Task-Specific Head Frozen:** For instance, freezing the `sentiment_head` while training BERT (if unfrozen) and the `category_head`. This scenario is less common for standard multi-task learning. If the frozen head's loss still contributes to the backward pass, BERT receives potentially conflicting update signals. If the frozen head's loss is ignored, it essentially becomes single-task fine-tuning for the category task. I would generally avoid this unless I had a specific reason, like having already perfected one task head.

**Transfer Learning Approach:**

In a scenario requiring transfer learning, such as adapting the model to a new domain:

1. **Choice of Pre-trained Model:** I would start with a robust base like `bert-base-uncased`, or preferably, a model pre-trained on text from the target domain (like BioBERT for medical text) if available, as this provides a more relevant starting point.
2. **Layers to Freeze/Unfreeze:** My approach involves two phases:
   * *Phase 1:* Freeze the entire pre-trained backbone. This aligns with the default setup in my code. Train only the newly initialized task-specific heads on the target domain data.
   * *Phase 2:* Unfreeze the backbone (potentially just the upper layers) and continue training all parts together, but using a significantly lower learning rate for the backbone layers compared to the heads.
3. **Rationale:** This strategy first leverages the extensive knowledge captured in the pre-trained model without disrupting it, allowing the task heads to adapt quickly. Then, it carefully fine-tunes the backbone to the specific nuances of the new domain and tasks, minimizing the risk of catastrophic forgetting by using a low learning rate. This aims to balance general knowledge transfer with task-specific specialization.

---

# **Task 4: Training Loop Implementation**

I implemented a training loop (`train.py`) for the `MultiTaskSentenceEncoder`.

* **Data:** I set up a PyTorch `Dataset` and `DataLoader` to handle hypothetical data, ensuring each batch contained text along with corresponding labels for both product category and sentiment.
* **Forward Pass:** The model takes tokenized text and outputs a dictionary with separate logits for the category and sentiment tasks.
* **MTL Loss:** The main consideration for multi-task learning here was the loss. I calculated the cross-entropy loss for each task separately. To combine them for the backward pass, I chose the simplest approach: summing the two losses (`total_loss = category_loss + sentiment_loss`). This gives equal weight to both tasks. More advanced methods exist, like using learned weights or uncertainty weighting, but summing them provides a straightforward baseline.
* **Metrics:** I tracked performance by calculating accuracy and F1-score separately for both the category and sentiment tasks during evaluation using `sklearn.metrics`.

---

# Sources:

https://arxiv.org/pdf/1908.10084

https://www.marqo.ai/course/introduction-to-sentence-transformers

https://huggingface.co/docs/transformers/en/model_doc/bert

https://fetch.com/brands
