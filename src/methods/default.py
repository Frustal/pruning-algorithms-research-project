import torch
import copy
from tqdm import tqdm
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total

def run(config, model, loaders, logger):
    device = torch.device(config['training']['device'])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9)
    
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Default Training for {config['training']['epochs']} epochs...")

    for epoch in range(1, config['training']['epochs'] + 1):
        train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_acc = evaluate(model, loaders['val'], device)
        
        print(f"Epoch {epoch} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir / f"{config['experiment_name']}_best.pth")

    # Final Evaluation on Test Set
    print("Evaluating best model on Test Set...")
    model.load_state_dict(best_wts)
    test_acc = evaluate(model, loaders['test'], device)
    
    logger.log({
        "epoch": config['training']['epochs'],
        "val_acc": best_acc,
        "test_acc": test_acc,
        "sparsity": 0.0,
        "params": sum(p.numel() for p in model.parameters())
    })
    print(f"Test Acc: {test_acc:.2%}")