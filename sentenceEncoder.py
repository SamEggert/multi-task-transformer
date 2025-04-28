from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style, init

# Initialize colorama
init()

class SentenceEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(SentenceEncoder, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
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
        
        return sentence_embeddings

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
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        sentence_embeddings = model(input_ids, attention_mask)
    
    return sentence_embeddings

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SentenceEncoder()
    model.to(device)
    
    # Example sentences
    texts = [
        "I thought the cheese was delicious",
        "The cheese was tasty",
        "Luke, I am your father!",
        "We're gonna need a bigger boat!"
    ]
    
    # Get sentence embeddings
    embeddings = encode_sentences(texts, model, device)
    
    # Example: Calculate similarity between sentences
    for i in range(len(texts) - 1):
        cos_sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[i+1].unsqueeze(0))
        similarity = cos_sim.item()
        
        # Choose color based on similarity
        if similarity > 0.8:
            color = Fore.GREEN
        elif similarity > 0.5:
            color = Fore.YELLOW
        else:
            color = Fore.RED
            
        print(f"{Fore.CYAN}Comparison {i + 1}:{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Sentence 1:{Style.RESET_ALL}    {texts[i]}")
        print(f"{Fore.BLUE}Sentence 2:{Style.RESET_ALL}    {texts[i+1]}")
        print(f"{Fore.MAGENTA}Cosine similarity between sentence embeddings:{Style.RESET_ALL}   {color}{similarity:.4f}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()

