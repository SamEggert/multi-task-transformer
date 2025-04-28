from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style, init

# Initialize colorama
init()

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

class MultiTaskSentenceEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(MultiTaskSentenceEncoder, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
        # Get BERT's hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Task A: Product Category Classification Head
        self.category_head = nn.Linear(hidden_size, len(PRODUCT_CATEGORIES))
        
        # Task B: Sentiment Analysis Head
        self.sentiment_head = nn.Linear(hidden_size, len(SENTIMENT_CATEGORIES))
        
    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask from [batch_size, seq_length] to [batch_size, seq_length, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum all tokens and account for padding tokens using attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean by dividing sum by the number of non-padding tokens
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Generate sentence embeddings using mean pooling
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Get predictions from both heads
        category_logits = self.category_head(sentence_embeddings)
        sentiment_logits = self.sentiment_head(sentence_embeddings)
        
        return {
            'embeddings': sentence_embeddings,
            'category_logits': category_logits,
            'sentiment_logits': sentiment_logits
        }

def encode_sentences(texts, model, device):
    # Tokenize the input texts
    encoded = model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get outputs
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    return outputs

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MultiTaskSentenceEncoder()
    model.to(device)
    
    # Example sentences
    texts = [
        "I thought the cheese was delicious",
        "The cheese was tasty",
        "Luke, I am your father!",
        "We're gonna need a bigger boat!"
    ]
    
    # Get model outputs
    outputs = encode_sentences(texts, model, device)
    
    # Get predictions
    category_probs = F.softmax(outputs['category_logits'], dim=1)
    sentiment_probs = F.softmax(outputs['sentiment_logits'], dim=1)
    
    # Get predicted categories and sentiments
    predicted_categories = torch.argmax(category_probs, dim=1)
    predicted_sentiments = torch.argmax(sentiment_probs, dim=1)
    
    # Print results
    for i, text in enumerate(texts):
        print(f"\n{Fore.CYAN}Text:{Style.RESET_ALL} {text}")
        print(f"{Fore.BLUE}Predicted Category:{Style.RESET_ALL} {PRODUCT_CATEGORIES[predicted_categories[i]]}")
        print(f"{Fore.BLUE}Predicted Sentiment:{Style.RESET_ALL} {SENTIMENT_CATEGORIES[predicted_sentiments[i]]}")
        
        # Print top 3 category probabilities
        top_cat_probs, top_cat_indices = torch.topk(category_probs[i], 3)
        print(f"{Fore.MAGENTA}Top 3 Category Probabilities:{Style.RESET_ALL}")
        for prob, idx in zip(top_cat_probs, top_cat_indices):
            print(f"  {PRODUCT_CATEGORIES[idx]}: {prob:.4f}")
        
        # Print sentiment probabilities
        print(f"{Fore.MAGENTA}Sentiment Probabilities:{Style.RESET_ALL}")
        for j, sentiment in enumerate(SENTIMENT_CATEGORIES):
            print(f"  {sentiment}: {sentiment_probs[i][j]:.4f}")

if __name__ == "__main__":
    main()
