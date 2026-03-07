import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
