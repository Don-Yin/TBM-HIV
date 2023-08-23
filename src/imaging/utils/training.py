from src.imaging.utils.logger import get_logger
from src.imaging.utils.general import device, save_checkpoint
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
import torch

# disable warnings from sklearn
import warnings

warnings.filterwarnings("ignore")


def get_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == labels).float().mean().item()
    return accuracy


def get_balanced_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    predicted_np = predicted.cpu().numpy()
    labels_np = labels.cpu().numpy()
    balanced_accuracy = balanced_accuracy_score(labels_np, predicted_np)
    return balanced_accuracy


def main_training_loop(
    model_combined,
    optimizer,
    criterion,
    epochs,
    dataset_train,
    dataset_valid,
    checkpoint_path,
    batch_size,
    logger_handler,
    epochs_min,
):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        # Train the model
        for i, train_batch in enumerate(dataloader_train):
            # --------------------[training step]--------------------
            model_combined.train()
            images, labels, clinical_data = train_batch
            images, labels, clinical_data = images.to(device), labels.to(device), clinical_data.to(device)
            batch_size, channel, x, y, z = images.shape

            optimizer.zero_grad()
            logits = model_combined(images, clinical_data)  # with a shape of batch_size * output_classes
            loss = criterion(logits, labels)

            train_accuracy = get_accuracy(logits, labels)
            train_accuracy_balanced = get_balanced_accuracy(logits, labels)

            logger_handler.log({"train_accuracy": train_accuracy})
            logger_handler.log({"train_accuracy_balanced": train_accuracy_balanced})
            logger_handler.log({"train_accuracy_balanced_moving": logger_handler.moving_average_get("train_accuracy_balanced")})

            loss.backward()
            optimizer.step()

            # train_loss = loss.item() / batch_size
            train_loss = loss.item()

            # --------------------[end training step]--------------------
            logger_handler.log({"train_loss": train_loss})

            # --------------------[validation step]--------------------
            if i % 3 == 0 or i == 0:
                valid_batch = next(iter(dataloader_valid))
                model_combined.eval()

                with torch.no_grad():
                    images, labels, clinical_data = valid_batch
                    images, labels, clinical_data = images.to(device), labels.to(device), clinical_data.to(device)

                    batch_size, channel, x, y, z = images.shape
                    logits = model_combined(images, clinical_data)
                    loss = criterion(logits, labels)

                    valid_accuracy = get_accuracy(logits, labels)
                    valid_accuracy_balanced = get_balanced_accuracy(logits, labels)

                    logger_handler.log({"valid_accuracy": valid_accuracy})
                    logger_handler.log({"valid_accuracy_balanced": valid_accuracy_balanced})
                    logger_handler.log(
                        {"valid_accuracy_balanced_moving": logger_handler.moving_average_get("valid_accuracy_balanced")}
                    )

                # valid_loss = loss.item() / batch_size
                valid_loss = loss.item()
                logger_handler.log({"valid_loss": valid_loss})

                save_checkpoint(model_combined, checkpoint_path)

                if (
                    logger_handler.moving_average_get("valid_accuracy_balanced_moving") > 0.90
                    or logger_handler.moving_average_get("valid_loss") < 0.05
                ) and epoch > epochs_min:
                    logger_handler.on_early_stop()
                    return

    logger_handler.on_end()
