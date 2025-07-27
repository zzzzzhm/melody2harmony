import choraleDataloader
import torch
import BiLSTMModel
class Pipeline:
    def __init__(self, model, dataset, vocab_size,epochs, batch_size=32, lr=1e-3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.vocab_size = vocab_size
    
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.total_step = len(self.dataloader) * epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay = 1e-5)
        self.schedule = CosineAnnealingLR(self.optimizer, T_max=self.total_step)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        self.model.train()
        best_loss = float('inf')
        patience = 5
        count = 0
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x) 
                loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.dataloader)
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'bestmusicLLaMA.pt')
                count = 0
            else:
                count += 1
            if count >= patience:
                print("Early stopping!")
                break