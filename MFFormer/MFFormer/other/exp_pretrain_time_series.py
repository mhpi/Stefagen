import os
import time
import warnings
import numpy as np
import pandas as pd

import torch

from MFFormer.data_provider.data_factory import data_provider
from MFFormer.exp.exp_basic import Exp_Basic
from MFFormer.utils.tools import adjust_learning_rate, visual  # EarlyStopping,

from MFFormer.utils.stats.metrics import cal_stations_metrics

warnings.filterwarnings('ignore')


class ExpPretrainTimeSeries(Exp_Basic):
    def __init__(self, args, config_dataset):
        super().__init__(args)

        configs = args
        self.configs = configs

        self.train_data, self.train_loader = data_provider(args=self.args, config_dataset=config_dataset, flag='train')
        self.scaler = self.train_data.scaler
        self.vali_data, self.vali_loader = data_provider(args=self.args, config_dataset=config_dataset, flag='val',
                                                         scaler=self.scaler)
        self.test_data, self.test_loader = data_provider(args=self.args, config_dataset=config_dataset, flag='test',
                                                         scaler=self.scaler)

    def compute_loss(self, batch_data_dict, output_dict):
        batch_y = batch_data_dict["batch_x"]
        outputs = output_dict["outputs_time_series"]
        target_stds = batch_data_dict["batch_target_std"]

        f_dim = 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y, target_stds=target_stds)
        return loss

    def train(self):

        train_data, train_loader = self.train_data, self.train_loader
        vali_data, vali_loader = self.vali_data, self.vali_loader

        # resume from checkpoint
        self.load_model_resume()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in np.arange(self.start_epoch, self.args.epochs):
            epoch += 1

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data_dict in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()

                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_dict = self.model(batch_data_dict)
                        loss = self.compute_loss(batch_data_dict, output_dict)
                        train_loss.append(loss.item())
                else:
                    output_dict = self.model(batch_data_dict)

                    loss = self.compute_loss(batch_data_dict, output_dict)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.model_optim)
                    scaler.update()
                else:
                    loss.backward()

                    # clip gradients
                    if self.args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

                    self.model_optim.step()

            self.save_model(self.checkpoints_dir, epoch, self.model, self.model_optim, self.epoch_train_loss_list,
                            self.epoch_val_loss_list)

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)

            # print the time of each epoch
            print("Epoch: {} loss: {:.3f} cost time: {:.3f}".format(epoch, np.average(train_loss),
                                                                    time.time() - epoch_time))

            # # update the dataset index
            # train_loader.dataset.get_batch_sample_index(True)

            self.epoch_train_loss_list.append(np.average(train_loss))
            self.plot_loss_curve(self.epoch_train_loss_list, None,
                                 os.path.join(self.pics_dir, "loss_curve_only_train.png"))

            if self.args.do_eval:
                # validation
                val_loss = self.vali(vali_data, vali_loader, self.criterion)
                self.epoch_val_loss_list.append(val_loss)
                # plot the loss curve
                self.plot_loss_curve(self.epoch_train_loss_list, self.epoch_val_loss_list,
                                     os.path.join(self.pics_dir, "loss_curve.png"))

        print("results saved in: ", os.path.abspath(self.results_dir))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data_dict in enumerate(vali_loader):

                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_dict = self.model(batch_data_dict)
                else:
                    output_dict = self.model(batch_data_dict)

                loss = self.compute_loss(batch_data_dict, output_dict)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test=0):
        test_data, test_loader = self.test_data, self.test_loader

        if test:
            print('loading model')
            self.load_model_resume()

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data_dict in enumerate(test_loader):

                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }

                batch_y = batch_data_dict["batch_x"]

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_dict = self.model(batch_data_dict)
                else:
                    output_dict = self.model(batch_data_dict)

                outputs = output_dict["outputs_time_series"]
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # result save
        preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
        trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)

        raw_date_len = len(pd.date_range(test_data.config_dataset.test_date_list[0],
                                         test_data.config_dataset.test_date_list[-1]))
        assert raw_date_len == preds_restored.shape[1]

        # rescale the data
        preds_rescale = test_data.inverse_transform(preds_restored, mean=test_data.scaler["x_mean"],
                                                    std=test_data.scaler["x_std"])
        trues_rescale = test_data.inverse_transform(trues_restored, mean=test_data.scaler["x_mean"],
                                                    std=test_data.scaler["x_std"])

        # plot the final forecasting
        for idx_feature in range(preds_rescale.shape[-1]):
            visual(trues_rescale[0, :1000, idx_feature], preds_rescale[0, :1000, idx_feature],
                   os.path.join(self.pics_dir,
                                'feature_{}.pdf'.format(self.configs.time_series_variables[idx_feature])))

        f = open(os.path.join(self.results_dir, "result_regression.txt"), 'a')
        # write the configs
        f.write(self.config_format(self.configs))

        # calculate the metrics for each feature
        for idx_feature in range(preds_rescale.shape[-1]):
            metrics_list = ["NSE", "KGE", "Corr"]
            metrics_dict = cal_stations_metrics(trues_rescale[:, :, idx_feature], preds_rescale[:, :, idx_feature],
                                                metrics_list)
            print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.configs.time_series_variables[idx_feature],
                                                                      np.nanmedian(metrics_dict["NSE"]),
                                                                      np.nanmedian(metrics_dict["KGE"]),
                                                                      np.nanmedian(metrics_dict["Corr"])))
            f.write('\n')
            f.write("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.configs.time_series_variables[idx_feature],
                                                                        np.nanmedian(metrics_dict["NSE"]),
                                                                        np.nanmedian(metrics_dict["KGE"]),
                                                                        np.nanmedian(metrics_dict["Corr"])))

        print("obs_mean: {:.3f}, obs_median: {:.7f}".format(np.nanmean(trues_rescale), np.nanmedian(trues_rescale)))
        print("pred_mean: {:.3f}, pred_median: {:.7f}".format(np.nanmean(preds_rescale), np.nanmedian(preds_rescale)))

        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(self.results_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.results_dir, 'pred.npy'), preds_rescale)
        np.save(os.path.join(self.results_dir, 'true.npy'), trues_rescale)

        return
