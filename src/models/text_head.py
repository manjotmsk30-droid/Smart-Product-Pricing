"""
Model 2: Text Head (DistilBERT → MLP Regressor)
Fine-tunable transformer for text-based price prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
# CHANGE: removed unused import get_linear_schedule_with_warmup
# REASON: it wasn't used; avoids lint warnings and tiny import overhead.
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd


class TextDataset(Dataset):
    """Dataset for text and target pairs."""
    
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = float(self.targets[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float32)
        }


class TextHead(nn.Module):
    """DistilBERT with MLP regression head."""
    
    def __init__(self, model_name='distilbert-base-uncased', dropout=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # CHANGE: infer hidden size from encoder config (was hard-coded 768).
        # REASON: keeps compatibility with any DistilBERT variant.
        hidden = self.bert.config.hidden_size
        
        # MLP head: hidden → 512 → 128 → 1
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get first token representation (DistilBERT uses the first token like [CLS])
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        # Regression
        logits = self.regressor(pooled)
        return logits.squeeze(-1)


class TextModelTrainer:
    """Trainer for text-based price prediction."""
    
    def __init__(self, 
                 model_name='distilbert-base-uncased',
                 max_length=128,
                 dropout=0.1,
                 device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TextHead(model_name, dropout).to(self.device)
        
        print(f"✓ Initialized TextHead on {self.device}")
    
    def train(self, 
              train_texts, train_targets,
              val_texts=None, val_targets=None,
              batch_size=32,
              epochs_frozen=5,
              epochs_unfrozen=2,
              lr_head=1e-3,
              lr_encoder=2e-5,
              weight_decay=0.01,
              gradient_clip=1.0):
        
        """
        Train in two phases: frozen encoder, then unfrozen.
        
        Args:
            train_texts: Training texts
            train_targets: Training targets (log1p space)
            val_texts: Validation texts
            val_targets: Validation targets
            batch_size: Batch size
            epochs_frozen: Epochs with frozen encoder
            epochs_unfrozen: Epochs with unfrozen last block
            lr_head: Learning rate for head
            lr_encoder: Learning rate for encoder
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            
        Returns:
            Training history
        """
        train_texts = pd.Series(train_texts).astype('string').fillna('').tolist()
        if val_texts is not None:
            val_texts = pd.Series(val_texts).astype('string').fillna('').tolist()

        # Create datasets
        train_dataset = TextDataset(
            train_texts, train_targets, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_texts is not None and val_targets is not None:
            val_dataset = TextDataset(
                val_texts, val_targets, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        history = {'train_loss': [], 'val_loss': []}
        
        # Phase A: Frozen encoder
        print("\nPhase A: Training with frozen encoder...")
        self._freeze_encoder()
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_head,
            weight_decay=weight_decay
        )
        
        for epoch in range(epochs_frozen):
            train_loss = self._train_epoch(train_loader, optimizer, gradient_clip)
            val_loss = self._validate(val_loader) if val_loader else None
            
            history['train_loss'].append(train_loss)
            # CHANGE: check for None, not truthiness.
            # REASON: 0.0 is a valid loss and should be recorded.
            if val_loss is not None:
                history['val_loss'].append(val_loss)
            
            print(
                f"Epoch {epoch+1}/{epochs_frozen} - "
                f"train_loss: {train_loss:.4f}" + 
                (f" - val_loss: {val_loss:.4f}" if val_loss is not None else "")
            )
        
        # Phase B: Unfreeze last transformer block
        if epochs_unfrozen > 0:
            print("\nPhase B: Fine-tuning with unfrozen last block...")
            self._unfreeze_last_block()
            
            # Discriminative learning rates
            optimizer = torch.optim.AdamW([
                {'params': self.model.bert.parameters(), 'lr': lr_encoder},
                {'params': self.model.regressor.parameters(), 'lr': lr_head}
            ], weight_decay=weight_decay)
            
            for epoch in range(epochs_unfrozen):
                train_loss = self._train_epoch(train_loader, optimizer, gradient_clip)
                val_loss = self._validate(val_loader) if val_loader else None
                
                history['train_loss'].append(train_loss)
                # CHANGE: same None-safe append for validation metrics.
                if val_loss is not None:
                    history['val_loss'].append(val_loss)
                
                print(
                    f"Epoch {epoch+1}/{epochs_unfrozen} - "
                    f"train_loss: {train_loss:.4f}" + 
                    (f" - val_loss: {val_loss:.4f}" if val_loss is not None else "")
                )
        
        return history
    
    def _train_epoch(self, train_loader, optimizer, gradient_clip):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader):
        """Validate on validation set."""
        if val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.MSELoss()(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, texts, batch_size=32):
        """Predict on new texts."""
        texts = pd.Series(texts).astype('string').fillna('').tolist()
        dataset = TextDataset(
            texts, 
            np.zeros(len(texts)),  # Dummy targets
            self.tokenizer,
            self.max_length
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def _freeze_encoder(self):
        """Freeze BERT encoder."""
        for param in self.model.bert.parameters():
            param.requires_grad = False
    
    def _unfreeze_last_block(self):
        """Unfreeze last transformer block only."""
        # CHANGE: ensure only the last block is trainable even if this is called directly.
        # REASON: makes the method idempotent and prevents accidental full fine-tuning.
        for i in range(len(self.model.bert.transformer.layer)):
            for p in self.model.bert.transformer.layer[i].parameters():
                p.requires_grad = False
        for p in self.model.bert.transformer.layer[-1].parameters():
            p.requires_grad = True
    
    def save(self, path):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()


if __name__ == "__main__":
    # Test TextHead
    print("Testing TextHead...")
    
    # Dummy data
    train_texts = [
        "Apple iPhone 13 Pro - 256GB",
        "Samsung Galaxy S21 - 128GB",
        "Organic Cotton T-Shirt",
    ] * 10
    train_targets = np.log1p(np.random.uniform(50, 500, len(train_texts)))
    
    # Initialize trainer
    trainer = TextModelTrainer(max_length=64)
    
    # Train
    history = trainer.train(
        train_texts, train_targets,
        batch_size=8,
        epochs_frozen=2,
        epochs_unfrozen=1
    )
    
    # Predict
    preds = trainer.predict(train_texts[:3])
    print(f"\nPredictions: {preds}")
    print("\n✓ TextHead test passed")
