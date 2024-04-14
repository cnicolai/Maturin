import time
import torch
from StraightNet import StraightNet
from dataset import get_crooked_scan_dataloaders


def training_loop(n_epochs, optimizer, model, train_loader, val_loader, loss_fn):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        start_time = time.time()
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs.squeeze(1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        with torch.no_grad():
            loss_val = 0.0
            for imgs, labels in val_loader:
                outputs = model(imgs)
                loss = loss_fn(outputs.squeeze(1), labels)
                loss_val += loss.item()

        print(
            f"Epoch {epoch}, train loss: {loss_train:.5f}, val loss: {loss_val:.3f}, time: {(time.time() - start_time):.5f} seconds"
        )


if __name__ == "__main__":
    device = torch.device("mps")
    batch_size = 8

    model = StraightNet().to(device)
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_crooked_scan_dataloaders(batch_size=batch_size)

    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
    )
