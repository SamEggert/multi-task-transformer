import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from multiTaskEncoder import (
    MultiTaskSentenceEncoder,
    PRODUCT_CATEGORIES,
    SENTIMENT_CATEGORIES
)
from transformers import BertTokenizer
from tqdm import tqdm

# Define a hypothetical dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, texts, category_labels, sentiment_labels, tokenizer, max_length=512):
        self.texts = texts
        self.category_labels = category_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        category_label = self.category_labels[idx]
        sentiment_label = self.sentiment_labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'category_label': torch.tensor(category_label, dtype=torch.long),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long)
        }

def create_hypothetical_data(num_samples=1000):
    """Create hypothetical data for demonstration purposes"""
    # Example texts
    texts = [
        "This baby formula is gentle on sensitive stomachs",
        "The new lipstick shade is absolutely stunning",
        "The wine has a rich, full-bodied flavor",
        "These cleaning supplies are very effective",
        "The frozen pizza was disappointing",
        "The pet food has high-quality ingredients",
        "The breakfast cereal is too sweet",
        "The beauty products are overpriced",
        "The beer selection is excellent",
        "The dairy products are fresh and delicious"
    ] * (num_samples // 10)  # Repeat to get desired number of samples

    # Random category labels (0 to 27)
    category_labels = np.random.randint(0, len(PRODUCT_CATEGORIES), size=num_samples)
    
    # Random sentiment labels (0 to 2)
    sentiment_labels = np.random.randint(0, len(SENTIMENT_CATEGORIES), size=num_samples)

    return texts, category_labels, sentiment_labels

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_category_loss = 0
    total_sentiment_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        category_labels = batch['category_label'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate losses
        category_loss = F.cross_entropy(outputs['category_logits'], category_labels)
        sentiment_loss = F.cross_entropy(outputs['sentiment_logits'], sentiment_labels)
        
        # Combined loss (equal weighting for both tasks)
        total_loss = category_loss + sentiment_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()

        total_category_loss += category_loss.item()
        total_sentiment_loss += sentiment_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'cat_loss': f'{category_loss.item():.4f}',
            'sent_loss': f'{sentiment_loss.item():.4f}'
        })

    return total_category_loss / len(dataloader), total_sentiment_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    category_preds = []
    category_labels = []
    sentiment_preds = []
    sentiment_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_label = batch['category_label'].to(device)
            sentiment_label = batch['sentiment_label'].to(device)

            outputs = model(input_ids, attention_mask)
            
            # Get predictions
            category_pred = torch.argmax(outputs['category_logits'], dim=1)
            sentiment_pred = torch.argmax(outputs['sentiment_logits'], dim=1)
            
            category_preds.extend(category_pred.cpu().numpy())
            category_labels.extend(category_label.cpu().numpy())
            sentiment_preds.extend(sentiment_pred.cpu().numpy())
            sentiment_labels.extend(sentiment_label.cpu().numpy())

    # Calculate metrics
    category_accuracy = accuracy_score(category_labels, category_preds)
    category_f1 = f1_score(category_labels, category_preds, average='weighted')
    
    sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
    sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted')

    return {
        'category_accuracy': category_accuracy,
        'category_f1': category_f1,
        'sentiment_accuracy': sentiment_accuracy,
        'sentiment_f1': sentiment_f1
    }

def main(save_checkpoints=False):
    print("\n=== Starting Training ===")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    model = MultiTaskSentenceEncoder()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create hypothetical data
    texts, category_labels, sentiment_labels = create_hypothetical_data(num_samples=50)
    
    # Create dataset and dataloader
    dataset = MultiTaskDataset(texts, category_labels, sentiment_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 3
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        category_loss, sentiment_loss = train_epoch(model, dataloader, optimizer, device)
        
        # Evaluate
        metrics = evaluate(model, dataloader, device)
        
        # Print results
        print(f"\nResults:")
        print(f"Category Loss: {category_loss:.4f}")
        print(f"Sentiment Loss: {sentiment_loss:.4f}")
        print(f"Category Accuracy: {metrics['category_accuracy']:.4f}")
        print(f"Category F1: {metrics['category_f1']:.4f}")
        print(f"Sentiment Accuracy: {metrics['sentiment_accuracy']:.4f}")
        print(f"Sentiment F1: {metrics['sentiment_f1']:.4f}")

        # Save checkpoint if enabled
        if save_checkpoints:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'category_loss': category_loss,
                'sentiment_loss': sentiment_loss,
                'metrics': metrics
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main(save_checkpoints=False)  # Default to False for demo purposes
