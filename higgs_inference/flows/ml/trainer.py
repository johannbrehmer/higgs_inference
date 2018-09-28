import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from higgs_inference.flows.various.utils import check_for_nans_in_parameters


class GoldDataset(torch.utils.data.Dataset):

    def __init__(self, theta, x, y=None, r_xz=None, t_xz=None):
        self.n = theta.shape[0]

        placeholder = torch.stack([tensor([0.]) for _ in range(self.n)])

        self.theta = theta
        self.x = x
        self.y = placeholder if y is None else y
        self.r_xz = placeholder if r_xz is None else r_xz
        self.t_xz = placeholder if t_xz is None else t_xz

        assert len(self.theta) == self.n
        assert len(self.x) == self.n
        assert len(self.y) == self.n
        assert len(self.r_xz) == self.n
        assert len(self.t_xz) == self.n

    def __getitem__(self, index):
        return (self.theta[index],
                self.x[index],
                self.y[index],
                self.r_xz[index],
                self.t_xz[index])

    def __len__(self):
        return self.n


def train_model(model,
                loss_functions,
                thetas, xs, ys=None, r_xzs=None, t_xzs=None,
                theta1=None,
                loss_weights=None,
                loss_labels=None,
                pre_loss_transformer=None,
                pre_loss_transform_coefficients=None,
                batch_size=64,
                trainer='adam',
                initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50,
                clip_gradient=1.,
                run_on_gpu=True,
                double_precision=False,
                validation_split=0.2, early_stopping=True, early_stopping_patience=None,
                learning_curve_folder=None, learning_curve_filename=None,
                verbose='some'):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Move model to device
    model = model.to(device, dtype)

    if pre_loss_transform_coefficients is not None:
        pre_loss_transform_coefficients = torch.tensor(pre_loss_transform_coefficients).to(device, dtype)

    logging.debug('Transform coefficients: %s', pre_loss_transform_coefficients)

    # Convert to Tensor
    thetas = torch.stack([tensor(i.astype(np.float), requires_grad=True) for i in thetas])
    xs = torch.stack([tensor(i.astype(np.float)) for i in xs])
    if ys is not None:
        ys = torch.stack([tensor(i.astype(np.float)) for i in ys])
    if r_xzs is not None:
        r_xzs = torch.stack([tensor(i.astype(np.float)) for i in r_xzs])
    if t_xzs is not None:
        t_xzs = torch.stack([tensor(i.astype(np.float)) for i in t_xzs])

    # Dataset
    dataset = GoldDataset(thetas, xs, ys, r_xzs, t_xzs)

    # Val split
    if validation_split is not None and validation_split <= 0.:
        validation_split = None

    # Train / validation split
    if validation_split is not None:
        assert 0. < validation_split < 1., 'Wrong validation split: {}'.format(validation_split)

        n_samples = len(dataset)
        indices = list(range(n_samples))
        split = int(np.floor(validation_split * n_samples))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            pin_memory=run_on_gpu
        )
        validation_loader = DataLoader(
            dataset,
            sampler=validation_sampler,
            batch_size=batch_size,
            pin_memory=run_on_gpu
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=run_on_gpu
        )

    # Optimizer
    if trainer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    elif trainer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
    else:
        raise ValueError('Unknown trainer {}'.format(trainer))

    # Early stopping
    early_stopping = early_stopping and (validation_split is not None) and (n_epochs > 1)
    early_stopping_best_val_loss = None
    early_stopping_best_model = None
    early_stopping_epoch = None

    # Loss functions
    n_losses = len(loss_functions)

    if loss_weights is None:
        loss_weights = [1.] * n_losses

    # Losses over training
    individual_losses_train = []
    individual_losses_val = []
    total_losses_train = []
    total_losses_val = []
    total_val_loss = None

    log_r = None

    # Verbosity
    n_epochs_verbose = None
    if verbose == 'all':  # Print output after every epoch
        n_epochs_verbose = 1
    elif verbose == 'some':  # Print output after 10%, 20%, ..., 100% progress
        n_epochs_verbose = max(int(round(n_epochs / 10, 0)), 1)

    logging.info('Starting training')

    # Loop over epochs
    for epoch in range(n_epochs):

        logging.debug('Epoch %s / %s', epoch + 1, n_epochs)

        # Training
        model.train()
        individual_train_loss = np.zeros(n_losses)
        total_train_loss = 0.0

        # Learning rate decay
        if n_epochs > 1:
            lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(epoch / (n_epochs - 1.))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Loop over batches
        for i_batch, (theta, x, y, r_xz, t_xz) in enumerate(train_loader):

            logging.debug('  Batch %s', i_batch + 1)

            # Put on device
            theta = theta.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype).view((-1,))
            try:
                r_xz = r_xz.to(device, dtype).view((-1,))
            except NameError:
                pass
            try:
                t_xz = t_xz.to(device, dtype)
            except NameError:
                pass
            if theta1 is not None:
                theta1_tensor = torch.tensor(theta1).to(device, dtype)
                theta1_tensor = theta1_tensor.view(1, -1).expand_as(theta)

            optimizer.zero_grad()

            # Evaluate model
            _, log_likelihood, score = model.log_likelihood_and_score(theta, x)
            if theta1 is not None:
                _, log_likelihood_theta1 = model.log_likelihood(theta1_tensor, x)
                log_r = log_likelihood - log_likelihood_theta1

            # Pre-loss transformation
            if pre_loss_transformer is not None:
                log_likelihood, log_r, score, y, r_xz, t_xz = pre_loss_transformer(
                    log_likelihood, log_r, score, y, r_xz, t_xz, coefficients=pre_loss_transform_coefficients
                )

            # Evaluate loss
            losses = [loss_function(log_likelihood, log_r, score, y, r_xz, t_xz) for loss_function in loss_functions]
            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_train_loss[i] += individual_loss.item()
            total_train_loss += loss.item()

            # Calculate gradient
            loss.backward()

            # Clip gradients
            if clip_gradient is not None:
                clip_grad_norm_(model.parameters(), clip_gradient)

            # Check for NaNs
            if check_for_nans_in_parameters(model):
                logging.warning('NaNs in parameters or gradients, stopping training!')
                break

            # Optimizer step
            optimizer.step()

        individual_train_loss /= len(train_loader)
        total_train_loss /= len(train_loader)

        total_losses_train.append(total_train_loss)
        individual_losses_train.append(individual_train_loss)

        # Validation
        if validation_split is None:
            if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
                logging.info('  Epoch %d: train loss %.2f (%s)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1]))
            continue

        # with torch.no_grad():
        model.eval()
        individual_val_loss = np.zeros(n_losses)
        total_val_loss = 0.0

        for i_batch, (theta, x, y, r_xz, t_xz) in enumerate(validation_loader):

            # Put on device
            theta = theta.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype).view((-1,))
            try:
                r_xz = r_xz.to(device, dtype).view((-1,))
            except NameError:
                pass
            try:
                t_xz = t_xz.to(device, dtype)
            except NameError:
                pass
            if theta1 is not None:
                theta1_tensor = torch.tensor(theta1).to(device, dtype)
                theta1_tensor = theta1_tensor.view(1, -1).expand_as(theta)

            # Evaluate model
            _, log_likelihood, score = model.log_likelihood_and_score(theta, x)
            if theta1 is not None:
                _, log_likelihood_theta1 = model.log_likelihood(theta1_tensor, x)
                log_r = log_likelihood - log_likelihood_theta1

            # Pre-loss transformation
            if pre_loss_transformer is not None:
                log_likelihood, log_r, score, y, r_xz, t_xz = pre_loss_transformer(
                    log_likelihood, log_r, score, y, r_xz, t_xz, coefficients=pre_loss_transform_coefficients
                )

            # Evaluate losses
            losses = [loss_function(log_likelihood, None, score, y, r_xz, t_xz) for loss_function in loss_functions]
            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_val_loss[i] += individual_loss.item()
            total_val_loss += loss.item()

        individual_val_loss /= len(validation_loader)
        total_val_loss /= len(validation_loader)

        total_losses_val.append(total_val_loss)
        individual_losses_val.append(individual_val_loss)

        # Early stopping: best epoch so far?
        if early_stopping:
            if early_stopping_best_val_loss is None or total_val_loss < early_stopping_best_val_loss:
                early_stopping_best_val_loss = total_val_loss
                early_stopping_best_model = model.state_dict()
                early_stopping_epoch = epoch

        # Print out information
        if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
            if early_stopping and epoch == early_stopping_epoch:
                logging.info('  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s) (*)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1],
                                total_losses_val[-1], individual_losses_val[-1]))
            else:
                logging.info('  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s)'
                             % (epoch + 1, total_losses_train[-1], individual_losses_train[-1],
                                total_losses_val[-1], individual_losses_val[-1]))

        # Early stopping: actually stop training
        if early_stopping and early_stopping_patience is not None:
            if epoch - early_stopping_epoch >= early_stopping_patience > 0:
                logging.info('No improvement for %s epochs, stopping training', epoch - early_stopping_epoch)
                break

    # Early stopping: back to best state
    if early_stopping:
        if early_stopping_best_val_loss < total_val_loss:
            logging.info('Early stopping after epoch %s, with loss %.2f compared to final loss %.2f',
                         early_stopping_epoch + 1, early_stopping_best_val_loss, total_val_loss)
            model.load_state_dict(early_stopping_best_model)
        else:
            logging.info('Early stopping did not improve performance')

    # Save learning curve
    if learning_curve_folder is not None and learning_curve_filename is not None:

        np.save(learning_curve_folder + '/loss_train_' + learning_curve_filename + '.npy', total_losses_train)
        if validation_split is not None:
            np.save(learning_curve_folder + '/loss_val_' + learning_curve_filename + '.npy', total_losses_val)

        if loss_labels is not None:
            individual_losses_train = np.array(individual_losses_train)
            individual_losses_val = np.array(individual_losses_val)

            for i, label in enumerate(loss_labels):
                np.save(
                    learning_curve_folder + '/loss_' + label + '_train' + learning_curve_filename + '.npy',
                    individual_losses_train[:, i]
                )
                if validation_split is not None:
                    np.save(
                        learning_curve_folder + '/loss_' + label + '_val' + learning_curve_filename + '.npy',
                        individual_losses_val[:, i]
                    )

    logging.info('Finished training')

    return total_losses_train, total_losses_val
