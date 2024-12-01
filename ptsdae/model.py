# model.py

from typing import Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptsdae.ae import AutoEncoder  # Assurez-vous que le chemin est correct selon votre structure de projet

def train_standard(
    dataset: torch.utils.data.Dataset,
    autoencoder: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[int, float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    """
    Fonction pour entraîner un auto-encodeur standard.

    :param dataset: Dataset d'entraînement
    :param autoencoder: Instance de l'auto-encodeur
    :param epochs: Nombre d'époques d'entraînement
    :param batch_size: Taille des batches
    :param optimizer: Optimiseur à utiliser
    :param scheduler: Scheduler pour ajuster le learning rate
    :param validation: Dataset de validation
    :param cuda: Utiliser CUDA si True
    :param sampler: Sampler pour le DataLoader
    :param silent: Supprimer les affichages si True
    :param update_freq: Fréquence des callbacks (en époques)
    :param update_callback: Fonction de callback avec (epoch, loss, val_loss)
    :param num_workers: Nombre de workers pour le DataLoader
    :param epoch_callback: Fonction de callback après chaque époque
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers is not None else 0,
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else 0,
        )
    else:
        validation_loader = None

    loss_function = nn.MSELoss()
    autoencoder.train()
    validation_loss_value = -1
    loss_value = 0

    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            # Gestion des tuples/lists de données
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) in [1, 2]:
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            # Passage à travers l'auto-encodeur
            output = autoencoder(batch)
            loss = loss_function(output, batch)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Mise à jour de la barre de progression
            data_iterator.set_postfix(
                epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
            )
        # Validation et callback
        if update_freq is not None and (epoch + 1) % update_freq == 0:
            if validation_loader is not None:
                autoencoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in validation_loader:
                        if (isinstance(val_batch, tuple) or isinstance(val_batch, list)) and len(val_batch) in [1, 2]:
                            val_batch = val_batch[0]
                        if cuda:
                            val_batch = val_batch.cuda(non_blocking=True)
                        val_output = autoencoder(val_batch)
                        loss = loss_function(val_output, val_batch)
                        val_loss += loss.item() * val_batch.size(0)
                val_loss /= len(validation_loader.dataset)
                validation_loss_value = val_loss
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.6f" % loss_value,
                    vls="%.6f" % validation_loss_value,
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                )
            if update_callback is not None:
                update_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                    validation_loss_value,
                )
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()


def predict_standard(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
) -> torch.Tensor:
    """
    Génère des prédictions (reconstructions ou encodages) à partir d'un dataset.

    :param dataset: Dataset d'évaluation
    :param model: Instance de l'auto-encodeur
    :param batch_size: Taille des batches
    :param cuda: Utiliser CUDA si True
    :param silent: Supprimer les affichages si True
    :param encode: Si True, retourne les encodages; sinon, les reconstructions
    :return: Tensor des prédictions
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        features.append(
            output.detach().cpu()
        )
    return torch.cat(features)
