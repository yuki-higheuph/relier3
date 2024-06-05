import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader

# データセットクラスを定義
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        label = self.data.loc[idx, 'label']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1 if label == 'positive' else 0)  # ポジティブなら1、ネガティブなら0
        }
        return inputs

# BERTモデルとトークナイザを読み込み
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # ポジティブ／ネガティブの2つのラベル

# データセットを読み込み
df = pd.read_csv('sentiment_data.csv')

# ハイパーパラメータ
BATCH_SIZE = 8
MAX_LENGTH = 32
LEARNING_RATE = 2e-5
EPOCHS = 3

# データローダーを作成
dataset = SentimentDataset(df, tokenizer, MAX_LENGTH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# オプティマイザーとロス関数を定義
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# トレーニングループ
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss:.4f}')

# トレーニングしたモデルを保存
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')
