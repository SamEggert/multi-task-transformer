import torch.nn as nn
from sentenceEncoder import SentenceEncoder

# Define the product categories
PRODUCT_CATEGORIES = [
    "Baby", "Baking", "Beauty", "Beer, Hard Cider & Seltzer", "Beverages",
    "Bread & Bakery", "Breakfast & Cereal", "Candy & Chocolate",
    "Canned Goods & Soups", "Cleaning & Home Improvement", "Condiments & Sauces",
    "Dairy & Refrigerated", "Deli", "Frozen", "Grocery", "Gum & Mints",
    "Health & Wellness", "Household", "Meat & Seafood", "Oral Care", "Outdoor",
    "Personal Care", "Pet", "Restaurants", "Retailers", "Snacks", "Spirits", "Wine"
]

# Define sentiment categories
SENTIMENT_CATEGORIES = ["negative", "neutral", "positive"]

class MultiTaskSentenceEncoder(SentenceEncoder):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super().__init__(pretrained_model)
        
        # Get BERT's hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Task A: Product Category Classification Head
        self.category_head = nn.Linear(hidden_size, len(PRODUCT_CATEGORIES))
        
        # Task B: Sentiment Analysis Head
        self.sentiment_head = nn.Linear(hidden_size, len(SENTIMENT_CATEGORIES))
        
        # Freeze BERT weights (redundant but explicit)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Get sentence embeddings from parent class
        sentence_embeddings = super().forward(input_ids, attention_mask)
        
        # Get predictions from both heads
        category_logits = self.category_head(sentence_embeddings)
        sentiment_logits = self.sentiment_head(sentence_embeddings)
        
        return {
            'embeddings': sentence_embeddings,
            'category_logits': category_logits,
            'sentiment_logits': sentiment_logits
        }
