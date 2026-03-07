import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            probs = torch.sigmoid(preds)
            predicted = (probs > 0.9).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    return avg_loss, precision, recall, f1, auc, all_probs, all_preds, all_targets
