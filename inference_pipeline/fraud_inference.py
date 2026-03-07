import torch

def predict_fraud(model, x_features, device="cpu"):
    """
    Runs inference on the PyTorch model for given preprocessed features.
    """
    x = torch.tensor(x_features, dtype=torch.float32).to(device)
    
    # If a single instance is passed without batch dimension
    if x.dim() == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().item()

    return prob
