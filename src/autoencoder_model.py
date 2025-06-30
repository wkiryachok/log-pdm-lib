import json
import os
import random
from io import StringIO
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray = None):
        if X is None:
            raise ValueError("X not defined!")
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class AutoEncoder(nn.Module):
    @staticmethod
    def get_activation_func(
        activation_func: Literal[
            "relu", "sigmoid", "gelu", "celu", "selu", "elu", "tanh"
        ] = "gelu",
    ):
        """Determine which non linearity is to be used for either encoder or decoder"""
        if activation_func == "relu":
            return nn.ReLU()
        if activation_func == "sigmoid":
            return nn.Sigmoid()
        if activation_func == "gelu":
            return nn.GELU()
        if activation_func == "celu":
            return nn.CELU()
        if activation_func == "selu":
            return nn.SELU()
        if activation_func == "elu":
            return nn.ELU()
        if activation_func == "tanh":
            return nn.Tanh()
        return None

    def __init__(
        self,
        feats_dim: int,
        encoder_sizes: list = None,
        decoder_sizes: list = None,
        activation_func: Literal[
            "relu", "sigmoid", "gelu", "celu", "selu", "elu", "tanh"
        ] = "gelu",
    ):
        """
        The idea behind is to firstly convolve input tensor with size (B, 1, N) to (B, C, 1).
        Thus, we are getting a stack of convolution filters with learned weights.
        Then, the reconstructed tensor (B, C, 1) is fed into 1D convolution block.
        Thus, we finally obtain a tensor of size (B, N, 1).

        Args:
            feats_dim (int): Number of features in the input tensor.
            encoder_sizes (list of int): Sizes of linear layers in the encoder. Defaults to [64, 32, 16, 8].
            decoder_sizes (list of int): Sizes of linear layers in the decoder. Defaults to [8, 16, 32, 64].
            activation_func (str): activation function name for either encoder or decoder. Defaults to GELU.
        """
        super().__init__()

        act_func = AutoEncoder.get_activation_func(activation_func)

        if encoder_sizes is None:
            encoder_sizes = [64, 32, 16, 8]
        if decoder_sizes is None:
            decoder_sizes = encoder_sizes[::-1]

        if len(encoder_sizes) >= 3:
            layers_encoder = [
                (nn.Linear(m, n), act_func, nn.LayerNorm(n))
                for m, n in zip(encoder_sizes[:-2], encoder_sizes[1:])
            ]
            layers_encoder = [item for pair in layers_encoder for item in pair]
            layers_encoder.append(nn.Linear(encoder_sizes[-2], encoder_sizes[-1]))

            layers_decoder = [
                (nn.Linear(m, n), act_func, nn.LayerNorm(n))
                for m, n in zip(decoder_sizes[:-2], decoder_sizes[1:])
            ]
            layers_decoder = [item for pair in layers_decoder for item in pair]
            layers_decoder.append(nn.Linear(decoder_sizes[-2], decoder_sizes[-1]))
        else:
            raise ValueError("Length of layer_sizes must be greater than 2")

        self.conv_in = nn.Conv1d(
            in_channels=1, out_channels=encoder_sizes[0], kernel_size=feats_dim
        )
        self.conv_out = nn.Conv1d(
            in_channels=encoder_sizes[0], out_channels=feats_dim, kernel_size=1
        )

        self.encoder = nn.Sequential(*layers_encoder)
        self.decoder = nn.Sequential(*layers_decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, N) -> (B, 1, N) -(conv)-> (B, C, 1) -> (B, C)
        B - batch_size
        N - n_features
        C - encoder_sizes[0]
        Then, encoder head (consists of linear layers) encodes obtained tensor to the latent tensor.

        Args:
            x (torch.Tensor): tensor to be encoded by model's encoder head

        Returns:
            torch.Tensor: input tensor's latent representation
        """
        x = self.conv_in(x.unsqueeze(1)).squeeze(-1)
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C) -> (B, C, 1) -(conv)-> (B, N, 1) -> (B, N)
        B - batch_size
        N - n_features
        C - encoder_sizes[0]

        Args:
            x (torch.Tensor): latent representation tensor to be decoded by model's decoder head

        Returns:
            torch.Tensor: reconstructed tensor
        """
        x = self.decoder(x)
        x = self.conv_out(x.unsqueeze(2)).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: reconstructed input tensor
        """
        x = self.encode(x)
        x = self.decode(x)
        return x


class Seq2SeqAutoEncoder(nn.Module):
    def __init__(
        self,
        rnn_type: Literal["gru", "lstm", "rnn"] = "gru",
        input_size: int = None,
        channels: int = 128,
        hidden_size: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Almoust the same as AutoEncoder model insead of encoder-decoder network.
        A classical Sequence2Sequence model with Gated Reccurent Unit (GRU) used for encoding and decoding.

        Args:
            input_size (int): Number of features in the input tensor. Defaults to 6.
            channels (int): Number of output channels in convolutional layer. Defaults to 128.
            hidden_size (int): Number of features in the GRU hidden tensor (latent dimension). Defaults to 8.
            num_layers (int): Number of stacked GRU's layers. Defaults to 4.
            dropout (float, optional): Dropout value in the GRU cell. Defaults to .1.
            bidirectional (bool, optional): Bidirectional type of RNN's. Defaults to False.
        """
        super().__init__()

        if bidirectional:
            multiplier = 2
        else:
            multiplier = 1

        self.conv_in = nn.Conv1d(
            in_channels=input_size, out_channels=channels, kernel_size=1
        )

        self.conv_out = nn.Conv1d(
            in_channels=multiplier * channels, out_channels=input_size, kernel_size=1
        )

        if rnn_type == "gru":
            self.encoder = nn.GRU(
                input_size=channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.decoder = nn.GRU(
                input_size=multiplier * hidden_size,
                hidden_size=channels,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.decoder = nn.LSTM(
                input_size=multiplier * hidden_size,
                hidden_size=channels,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "rnn":
            self.encoder = nn.RNN(
                input_size=channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.decoder = nn.RNN(
                input_size=multiplier * hidden_size,
                hidden_size=channels,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        First, model converts input tensor with dimensionality (B, 1, N) to (B, C, 1) (the last dimension then squeezed)
        in order to obtain useful representation of the input tensor.
        Then, encoder head (consists GRU layer) encodes obtained tensor to the latent tensor.
        (B, W, N) -> (B, N, W) -(conv)-> (B, W, C) -> (B, W, H)
        B - batch_size
        W - window_size
        N - n_features
        C - channels
        H - hidden_size

        Args:
            x (torch.Tensor): tensor to be encoded by model's encoder head

        Returns:
            torch.Tensor: input tensor's latent representation
        """
        x = self.conv_in(x.transpose(2, 1)).transpose(2, 1)
        x, _ = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        First, decoder (GRU layer) decodes latent tensor (B, L) to (B, C).
        Then, model convers (B, C, 1) -> (B, N, 1) (the last dimension then squeezed)

        Args:
            x (torch.Tensor): latent representation tensor to be decoded by model's decoder head

        Returns:
            torch.Tensor: reconstructed tensor
        """
        x, _ = self.decoder(x)
        x = self.conv_out(x.transpose(2, 1)).transpose(2, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: reconstructed input tensor
        """
        x = self.encode(x)
        x = self.decode(x)
        return x


class AEModel:
    """Класс, реализующий несколько разновидностей автоэнкодеров.
    Этот класс имеет два основных метода, доступных для пользователя: fit and predict.
    """

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __init__(
        self,
        method: Literal["simple_ae", "seq2seq_ae"] = "simple_ae",
        n_features: int = None,
        scale_model_name: Literal[
            "StandardScaler", "MinMaxScaler", "RobustScaler"
        ] = "StandardScaler",
        batch_size: int = 64,
        encoder_sizes: list = None,
        decoder_sizes: list = None,
        activation_func: Literal[
            "relu", "sigmoid", "gelu", "celu", "selu", "elu", "tanh"
        ] = "gelu",
        window_size: int = 32,
        padding: bool = True,
        rnn_type: Literal["gru", "lstm", "rnn"] = "gru",
        channels: int = 64,
        hidden_size: int = 8,
        num_layers: int = 4,
        dropout: float = 0,
        bidirectional: bool = False,
        random_state: int = 42,
    ):
        self.method = method
        self.scale_model_name = scale_model_name
        self.batch_size = batch_size
        self.scale_model = AEModel.get_new_scaler(self.scale_model_name)
        self.n_features = n_features
        # simple_ae parameters
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.activation_func = activation_func
        # seq2seq_ae parameters
        self.window_size = window_size
        self.padding = padding
        self.rnn_type = rnn_type
        self.channels = channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.ucl = None
        self.residuals_thresholds = pd.DataFrame()

        AEModel.set_seed(random_state)
        if self.method == "simple_ae":
            self.ae_model = AutoEncoder(
                feats_dim=self.n_features,
                encoder_sizes=self.encoder_sizes,
                decoder_sizes=self.decoder_sizes,
                activation_func=self.activation_func,
            )
        elif self.method == "seq2seq_ae":
            self.ae_model = Seq2SeqAutoEncoder(
                rnn_type=self.rnn_type,
                input_size=self.n_features,
                channels=self.channels,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    @staticmethod
    def get_new_scaler(
        scale_model_name: Literal[
            "StandardScaler", "MinMaxScaler", "RobustScaler"
        ] = "StandardScaler",
    ) -> object:
        """Возвращает пустой скейлер заданного типа

        Parameters
        ----------
        scale_model_name: str
            Название скейлера

        Returns
        -------
        scale_model: object
            Пустой скейлер
        """
        if scale_model_name == "StandardScaler":
            scale_model = StandardScaler()
        elif scale_model_name == "MinMaxScaler":
            scale_model = MinMaxScaler()
        elif scale_model_name == "RobustScaler":
            scale_model = RobustScaler()
        return scale_model

    @staticmethod
    def get_loss_func(
        loss_func_name: Literal["l1loss", "mseloss", "huberloss"] = "huberloss",
    ) -> None:
        """Determine which loss function is to be used for neural network learning"""
        loss_func_name = loss_func_name.lower()
        if loss_func_name == "l1loss":
            return nn.L1Loss()
        if loss_func_name == "mseloss":
            return nn.MSELoss()
        if loss_func_name == "huberloss":
            return nn.HuberLoss()
        return None

    def create_sequences(self, data: pd.DataFrame, verbose: bool = False) -> list:
        if self.window_size < 2:
            raise ValueError("window_size must be greater than 1")
        index = data.index
        ts_step = (index[1:] - index[:-1]).min()
        mask = list(index[1:] - index[:-1])
        mask.append(-1)

        indexes_ = []
        indexes_tmp = [index[0]]
        for i in range(len(mask)):
            if mask[i] == ts_step:
                indexes_tmp.append(index[i + 1])
            else:
                indexes_.append(indexes_tmp)
                if mask[i] != -1:
                    indexes_tmp = [index[i + 1]]

        seq = []
        indexes = []
        for indexes_tmp in indexes_:
            if (len(indexes_tmp) < self.window_size) & self.padding:
                indexes_tmp = np.pad(
                    indexes_tmp,
                    pad_width=(self.window_size - len(indexes_tmp), 0),
                    mode="edge",
                )
            if len(indexes_tmp) >= self.window_size:
                for i in range(len(indexes_tmp) - self.window_size + 1):
                    indexes.extend(indexes_tmp[i : i + self.window_size])
                    seq.append(
                        data.loc[indexes_tmp[i : i + self.window_size], :].values
                    )

        indexes = np.array(indexes)
        if verbose:
            print(
                f"Coverage of the dataset: {len(np.unique(indexes)) / len(index) * 100:.1f} %"
            )
        return seq, indexes

    def fit(
        self,
        train_dataset: pd.DataFrame,
        valid_dataset: pd.DataFrame = None,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        warmup: int = 0.2,
        loss_function: Literal["l1loss", "mseloss", "huberloss"] = "huberloss",
        device: Literal["cuda", "cpu"] = "cuda",
        eps_early_stop: float = 1e-5,
        stopping_rounds: int = 10,
        verbose: bool = False,
        random_state: int = 42,
    ) -> None:
        """Функция обучения автоэнкодера

        Parameters
        ----------
        dataset: pd.DataFrame
            Обучающий датасет
        ...
        verbose: bool
            Предоставляет возможность выводить информацию о расчетах в логи
        """
        AEModel.set_seed(random_state)
        train_dataset_scaled = pd.DataFrame(
            self.scale_model.fit_transform(train_dataset.values),
            index=train_dataset.index,
        )
        if valid_dataset is not None:
            valid_dataset_scaled = pd.DataFrame(
                self.scale_model.transform(valid_dataset.values),
                index=valid_dataset.index,
            )

        self.n_features = train_dataset.shape[1]

        if self.method == "seq2seq_ae":
            train_dataset_scaled, _ = self.create_sequences(
                train_dataset_scaled, verbose
            )
            if valid_dataset is not None:
                valid_dataset_scaled, _ = self.create_sequences(
                    valid_dataset_scaled, verbose
                )
        else:
            train_dataset_scaled = train_dataset_scaled.values
            if valid_dataset is not None:
                valid_dataset_scaled = valid_dataset_scaled.values

        train_dataset_torch = CustomDataset(train_dataset_scaled)
        train_dataloader = DataLoader(
            train_dataset_torch, batch_size=self.batch_size, shuffle=False
        )
        if valid_dataset is not None:
            valid_dataset_torch = CustomDataset(valid_dataset_scaled)
            valid_dataloader = DataLoader(
                valid_dataset_torch, batch_size=self.batch_size, shuffle=False
            )

        ae_model_device = self.ae_model.to(device)

        optimizer = torch.optim.Adam(ae_model_device.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=warmup,
        )

        loss_func_torch = AEModel.get_loss_func(loss_function)

        train_losses = []
        if valid_dataset is not None:
            valid_losses = []
        lrs = []
        for epoch in range(epochs):
            total_loss = 0.0
            ae_model_device.train()
            for _, batch in enumerate(train_dataloader):
                X_ = batch.to(device, dtype=torch.float32)
                outs = ae_model_device.forward(X_)
                loss = loss_func_torch(outs, X_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])
                total_loss += loss.item()
            average_loss = total_loss / len(train_dataloader)
            train_losses.append(average_loss)

            if valid_dataset is not None:
                total_loss = 0.0
                ae_model_device.eval()
                with torch.no_grad():
                    for _, batch in enumerate(valid_dataloader):
                        X_ = batch.to(device, dtype=torch.float32)
                        outs = ae_model_device.forward(X_)
                        loss = loss_func_torch(outs, X_)
                        total_loss += loss.item()
                average_loss = total_loss / len(valid_dataloader)
                valid_losses.append(average_loss)

            if verbose & ((epoch + 1) % 10 == 0):
                if valid_dataset is None:
                    print(
                        f"Epoch: {epoch + 1}/{epochs}, training loss = {train_losses[-1]:.6f}"
                    )
                else:
                    print(
                        f"Epoch: {epoch + 1}/{epochs}, training loss = {train_losses[-1]:.6f}, validation loss = {valid_losses[-1]:.6f}"
                    )

            if epoch > (stopping_rounds + 1):
                delta_loss = np.array(
                    train_losses[-(stopping_rounds + 1) : -1]
                ) - np.array(train_losses[-stopping_rounds:])
                if np.mean(np.abs(delta_loss)) < eps_early_stop:
                    if verbose:
                        print("Epoch number =", epoch)
                    break

        if verbose:
            plt.figure(figsize=(7, 5))
            plt.plot(train_losses, label="Training loss")
            if valid_dataset is not None:
                plt.plot(valid_losses, label="Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss value")
            plt.title("Losses")
            plt.legend()
            plt.show()

            plt.figure(figsize=(7, 5))
            plt.plot(lrs)
            plt.xlabel("Batch")
            plt.ylabel("Learning rate")
            plt.title("Learning rate")
            plt.show()

            print("Final training loss =", train_losses[-1])
            if valid_dataset is not None:
                print("Final validation loss =", valid_losses[-1])

    def predict(
        self,
        data: pd.DataFrame,
        weights_tags: pd.Series = None,
        return_health_index: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        verbose: bool = False,
    ) -> pd.DataFrame:
        data_df = pd.DataFrame(
            self.scale_model.transform(data.values), index=data.index
        )

        if self.method == "seq2seq_ae":
            values, index = self.create_sequences(data_df, verbose)
        else:
            values = data_df.values

        dataset = CustomDataset(values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        ae_model_device = self.ae_model.to(device)
        ae_model_device.eval()

        if self.method == "seq2seq_ae":
            results = pd.DataFrame(
                0, index=np.unique(index), columns=data.columns, dtype="float32"
            )
            counter = pd.Series(0, index=np.unique(index))
            i = 0
        else:
            predictions = []

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                x_values_batch = batch.to(device, dtype=torch.float32)
                outs = ae_model_device.forward(x_values_batch)
                if self.method == "seq2seq_ae":
                    outs = outs.reshape(
                        batch.shape[0] * self.window_size, self.n_features
                    )
                outs = self.scale_model.inverse_transform(outs.detach().cpu().numpy())
                if self.method == "seq2seq_ae":
                    indexes = index[i : (i + len(outs))]
                    i += len(outs)
                    if batch.shape[0] == 1:
                        results.loc[indexes, :] += outs
                        counter[indexes] += 1
                    else:
                        for j, indexes_ in enumerate(
                            np.array_split(indexes, len(indexes) / self.window_size)
                        ):
                            results.loc[indexes_, :] += outs[
                                (j * self.window_size) : ((j + 1) * self.window_size)
                            ]
                            counter[indexes_] += 1
                else:
                    predictions.append(outs)

        if self.method == "seq2seq_ae":
            results = results.div(counter, axis=0)
        else:
            results = pd.DataFrame(
                np.concatenate(predictions, axis=0),
                index=data.index,
                columns=data.columns,
            )

        if return_health_index:
            health_index, health_index_weigths = self.__calc_health_index(
                data, results, weights_tags
            )
            return {
                "estimate": results,
                "health_index": health_index,
                "health_index_weigths": health_index_weigths,
            }
        else:
            return results

    def __calc_health_index(
        self,
        data: pd.DataFrame,
        data_est: pd.DataFrame,
        weights_tags: pd.Series = None,
    ) -> None:
        """
        Функция расчета индекса технического состояния агрегата

        Parameters
        ----------
        data: pd.DataFrame
            Входная матрица состояний системы
        data_est: pd.DataFrame
            Оцененная матрица состояний системы
        weights_tags: pd.Series
            Вектор весов тегов в индексе технического состояния

        Returns
        -------
        health_index: pd.Series
            Индекс технического состояния агрегата
        health_index_weights: pd.DataFrame
            Вклад тегов в индекс технического состояния агрегата
        """
        residuals = self.scale_model.transform(
            data.values
        ) - self.scale_model.transform(data_est.values)
        if weights_tags is not None:
            if any(data.columns != weights_tags.index):
                raise ValueError(
                    "Columns in x_matrix must be equals indexes in weights_tags"
                )
            residuals = residuals * weights_tags.values

        health_index = pd.Series(
            np.linalg.norm(residuals, 2, axis=1) / np.sqrt(residuals.shape[1]),
            index=data.index,
        )

        health_index_weights = pd.DataFrame(
            np.full(residuals.shape, np.nan),
            index=data.index,
            columns=data.columns,
        )
        ind = ((health_index > 1e-6) & (~health_index.isna())).values
        health_index_weights.loc[ind, :] = (
            np.abs(residuals[ind]) / np.abs(residuals[ind]).sum(axis=1)[:, np.newaxis]
        )
        return health_index, health_index_weights

    def latent_space_vis(self, data: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
        values = self.scale_model.transform(data.values)

        dataset = CustomDataset(values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        ae_model_device = self.ae_model.to(device)
        predictions = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                x_values_batch = batch.to(device, dtype=torch.float32)
                outs = ae_model_device.encode(x_values_batch)
                predictions.append(outs.detach().cpu().numpy())

        dataframe_ae = pd.DataFrame(
            np.concatenate(predictions, axis=0),
            index=data.index,
            columns=[f"column_{i + 1}" for i in range(self.encoder_sizes[-1])],
        )

        return dataframe_ae

    def calc_health_index_threshold(
        self,
        valid_dataset: pd.DataFrame,
        weights_tags: pd.Series = None,
        health_index_quantile: float = 0.99,
        device: Literal["cuda", "cpu"] = "cpu",
    ):
        """
        Функция расчета трешхолда для индекса технического состояния

        Parameters
        ----------
        valid_dataset: pd.DataFrame
            Валидационный датасет
        weights_tags: pd.Series
            Вектор весов тегов в индексе технического состояния
        health_index_quantile: float
            Квантиль для расчета трешхолда индекса технического состояния

        Returns
        -------
        health_index_threshold: float
            Трешхолд для оценки макс. допустимого значения индекса технического состояния
        """
        valid_estimated = self.predict(valid_dataset, weights_tags, device=device)
        valid_residuals = self.scale_model.transform(
            valid_dataset.loc[valid_estimated.index].values
        ) - self.scale_model.transform(valid_estimated.values)

        if weights_tags is not None:
            if any(valid_dataset.columns != weights_tags.index):
                raise ValueError(
                    "Columns in dataset must be equals indexes in weights_tags"
                )
            valid_residuals = valid_residuals * weights_tags.values

        self.ucl = np.quantile(
            np.linalg.norm(valid_residuals, 2, axis=1)
            / np.sqrt(valid_dataset.shape[1]),
            health_index_quantile,
        )

    def calc_residuals_thresholds(
        self,
        valid_dataset: pd.DataFrame,
        device: Literal["cuda", "cpu"] = "cpu",
        residuals_quantile: float = 0.01,
    ) -> pd.DataFrame:
        """
        Функция расчета трешхолда для невязок

        Parameters
        ----------
        valid_dataset: pd.DataFrame
            Валидационный датасет
        residuals_quantile: float
            Квантиль для расчета трешхолдов для невязок

        Returns
        -------
        residuals_thresholds: pd.DataFrame
            Трешхолды для оценки макс. и мин. допустимых значений невязок
        """
        valid_estimated = self.predict(valid_dataset, device=device)
        valid_residuals = valid_dataset - valid_estimated

        self.residuals_thresholds = pd.DataFrame(
            [
                valid_residuals.quantile(1 - residuals_quantile),
                valid_residuals.quantile(residuals_quantile).abs(),
            ]
        ).T
        self.residuals_thresholds.index = valid_dataset.columns
        self.residuals_thresholds.columns = [
            "Residual (Positive)",
            "Residual (Negative)",
        ]

    @staticmethod
    def serialize_scaler(obj: object) -> str:
        attrs = {}
        for k, v in obj.__dict__.items():
            if k[-1:] != "_":
                continue
            if isinstance(v, np.ndarray):
                attrs[k] = v.tolist()
            if isinstance(v, int):
                attrs[k] = v
        return json.dumps(attrs)

    @staticmethod
    def deserialize_scaler(scaler_model: object, json_attrs: str) -> object:
        attrs = json.loads(json_attrs)
        for k, v in attrs.items():
            if k[-1:] != "_":
                continue
            if isinstance(v, list):
                setattr(scaler_model, k, np.array(v))
            else:
                setattr(scaler_model, k, v)
        return scaler_model

    def save_model(self, path_to_model: str):
        ae_model_json = {
            "method": self.method,
            "n_features": self.n_features,
            "scale_model_name": self.scale_model_name,
            "scale_model": AEModel.serialize_scaler(self.scale_model),
            "batch_size": self.batch_size,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "activation_func": self.activation_func,
            "rnn_type": self.rnn_type,
            "window_size": self.window_size,
            "padding": self.padding,
            "channels": self.channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "ucl": self.ucl,
            "residuals_thresholds": self.residuals_thresholds.to_json(),
        }
        ae_model_json_path = os.path.join(path_to_model, "ae_model.json")
        with open(ae_model_json_path, mode="w", encoding="utf-8") as opened_file:
            json.dump(ae_model_json, opened_file, indent=True)

        torch.save(
            self.ae_model.state_dict(), os.path.join(path_to_model, "ae_model.pth")
        )

    @staticmethod
    def load_model(path_to_model: str):
        ae_model_json_path = os.path.join(path_to_model, "ae_model.json")
        with open(ae_model_json_path, mode="r", encoding="utf-8") as opened_file:
            ae_model_json = json.load(opened_file, parse_int=int)

        ae_model = AEModel(
            method=ae_model_json["method"],
            n_features=ae_model_json["n_features"],
            scale_model_name=ae_model_json["scale_model_name"],
            batch_size=ae_model_json["batch_size"],
            encoder_sizes=ae_model_json["encoder_sizes"],
            decoder_sizes=ae_model_json["decoder_sizes"],
            activation_func=ae_model_json["activation_func"],
            rnn_type=ae_model_json["rnn_type"],
            window_size=ae_model_json["window_size"],
            padding=ae_model_json["padding"],
            channels=ae_model_json["channels"],
            hidden_size=ae_model_json["hidden_size"],
            num_layers=ae_model_json["num_layers"],
            dropout=ae_model_json["dropout"],
            bidirectional=ae_model_json["bidirectional"],
        )

        ae_model.scale_model = AEModel.deserialize_scaler(
            ae_model.scale_model, ae_model_json["scale_model"]
        )
        ae_model.ucl = ae_model_json["ucl"]
        ae_model.residuals_thresholds = pd.read_json(
            StringIO(ae_model_json["residuals_thresholds"])
        )

        if ae_model.method == "simple_ae":
            ae_model.ae_model = AutoEncoder(
                feats_dim=ae_model.n_features,
                encoder_sizes=ae_model.encoder_sizes,
                decoder_sizes=ae_model.decoder_sizes,
                activation_func=ae_model.activation_func,
            )
        elif ae_model.method == "seq2seq_ae":
            ae_model.ae_model = Seq2SeqAutoEncoder(
                rnn_type=ae_model.rnn_type,
                input_size=ae_model.n_features,
                channels=ae_model.channels,
                hidden_size=ae_model.hidden_size,
                num_layers=ae_model.num_layers,
                dropout=ae_model.dropout,
                bidirectional=ae_model.bidirectional,
            )

        ae_model.ae_model.load_state_dict(
            torch.load(os.path.join(path_to_model, "ae_model.pth"), weights_only=False)
        )
        ae_model.ae_model.eval()

        return ae_model
