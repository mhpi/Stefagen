import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

from MFFormer.data_provider.data_factory import data_provider
from MFFormer.exp.exp_basic import Exp_Basic
from MFFormer.utils.tools import adjust_learning_rate, visual  # EarlyStopping,

from MFFormer.utils.stats.metrics import cal_stations_metrics

warnings.filterwarnings('ignore')


class ExpPretrainHydroDL(Exp_Basic):
    def __init__(self, args, config_dataset):
        super().__init__(args, config_dataset)

        configs = args
        self.configs = configs

        # set weight for loss
        weights_time_series_variables = torch.ones(len(args.time_series_variables)) * 0.5
        weights_time_series_variables[-1] = 1.0
        # self.weights_time_series_variables = torch.nn.Parameter(weights_time_series_variables)
        # self.loss_weights_time_series = torch.nn.Parameter(torch.tensor([1.0]))
        # self.loss_weights_static = torch.nn.Parameter(torch.tensor([0.5]))
        self.weights_time_series_variables = weights_time_series_variables
        self.loss_weights_time_series = torch.tensor([1.0])
        self.loss_weights_static = torch.tensor([0.5])
        initial_ratio_categorical = len(args.static_variables_category) / len(args.static_variables)
        initial_ratio_numerical = 1 - initial_ratio_categorical
        self.loss_weights_static_numerical = torch.nn.Parameter(torch.tensor([initial_ratio_numerical]))
        self.loss_weights_static_categorical = torch.nn.Parameter(torch.tensor([initial_ratio_categorical]))

    def compute_loss(self, batch_data_dict, output_dict):
        batch_y = batch_data_dict["batch_x"]
        batch_c = batch_data_dict["batch_c"]
        outputs_time_series = output_dict["outputs_time_series"]
        outputs_static = output_dict["outputs_static"]
        target_stds = batch_data_dict["batch_target_std"]
        masked_time_series_index = output_dict["masked_time_series_index"]
        masked_static_index = output_dict["masked_static_index"]
        masked_missing_time_series_index = output_dict["masked_missing_time_series_index"]
        masked_missing_static_index = output_dict["masked_missing_static_index"]

        # remove masked_missing_index from masked_index using torch
        masked_time_series_index = masked_time_series_index & (~masked_missing_time_series_index)
        masked_static_index = masked_static_index & (~masked_missing_static_index)

        outputs_time_series = outputs_time_series[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        batch_c = batch_c.to(self.device)

        # weighted
        weighted_outputs = outputs_time_series * self.weights_time_series_variables.to(outputs_time_series.device)

        loss_time_series = self.criterion(weighted_outputs[masked_time_series_index],
                                          batch_y[masked_time_series_index], target_stds=target_stds)

        # calculate numerical and categorical loss
        if len(self.configs.static_variables_category) > 0:

            ratio_categorical = len(self.configs.static_variables_category) / len(self.configs.static_variables)
            ratio_numerical = 1 - ratio_categorical

            static_variables_dec_index_start = output_dict['static_variables_dec_index_start']
            static_variables_dec_index_end = output_dict['static_variables_dec_index_end']

            loss_target_categorical = 0
            total_categorical_indices_pred = []
            for ndx_static_variable_category, static_variable_category in enumerate(
                    self.configs.static_variables_category):
                indices_static_variable_category = self.configs.static_variables.index(static_variable_category)

                categorical_indices_pred = range(static_variables_dec_index_start[indices_static_variable_category],
                                                 static_variables_dec_index_end[indices_static_variable_category])
                categorical_indices_obs = [indices_static_variable_category]

                categorical_masked_static_index_obs = masked_static_index[:, categorical_indices_obs]  # [batch_size, 1]
                # broadcast to [batch_size, num_categorical]
                categorical_masked_static_index_pred = categorical_masked_static_index_obs.repeat(1,
                                                                                                  len(categorical_indices_pred))

                sub_loss_target_categorical = self.criterion_category(
                    outputs_static[:, categorical_indices_pred][categorical_masked_static_index_pred].reshape(-1,
                                                                                                              len(categorical_indices_pred)),
                    batch_c[:, categorical_indices_obs][categorical_masked_static_index_obs].long())

                loss_target_categorical += sub_loss_target_categorical
                total_categorical_indices_pred += categorical_indices_pred

            numerical_indices_pred = [x for x in range(outputs_static.shape[-1]) if
                                      x not in total_categorical_indices_pred]
            numerical_indices_obs = [self.configs.static_variables.index(x) for x in self.configs.static_variables if
                                     x not in self.configs.static_variables_category]

            numerical_masked_static_index_obs = masked_static_index[:, numerical_indices_obs]
            numerical_masked_static_index_pred = numerical_masked_static_index_obs

            loss_target_numerical = self.criterion(
                outputs_static[:, numerical_indices_pred][numerical_masked_static_index_pred],
                batch_c[:, numerical_indices_obs][numerical_masked_static_index_obs],
                target_stds=target_stds)
            # loss_static = loss_target_numerical * ratio_numerical + loss_target_categorical * ratio_categorical
            loss_static = loss_target_numerical * self.loss_weights_static_numerical.to(
                loss_target_numerical.device) + loss_target_categorical * self.loss_weights_static_categorical.to(
                loss_target_categorical.device)
        else:
            loss_static = self.criterion(outputs_static[masked_static_index], batch_c[masked_static_index],
                                         target_stds=target_stds)
        # loss = loss_time_series * self.configs.ratio_time_series + loss_static * self.configs.ratio_static
        loss = loss_time_series * self.loss_weights_time_series.to(
            loss_time_series.device) + loss_static * self.loss_weights_static.to(loss_static.device)
        return loss

    def train(self):

        train_data, train_loader = self.train_data, self.train_loader
        vali_data, vali_loader = self.vali_data, self.vali_loader

        # resume from checkpoint
        self.load_model_resume()

        # load the data
        train_x_norm = self.train_data.raw_x
        static_data_norm = self.train_data.raw_c

        def random_index(num_grid, num_time, dimSubset, train_warmup=0):
            bs, seq_len = dimSubset
            slice_grid = np.random.randint(0, num_grid, [bs])
            slice_time = np.random.randint(train_warmup, num_time - seq_len, [bs])
            return slice_grid, slice_time

        def select_subset(x, slice_grid, slice_time, seq_len, c=None, train_warmup=0):
            num_time = x.shape[1]
            if x.shape[0] == len(slice_grid):  # hack
                slice_grid = np.arange(0, len(slice_grid))  # hack
            if num_time <= seq_len:
                slice_time.fill(0)

            time_idx = slice_time[:, None] + np.arange(-train_warmup, seq_len)[None, :]
            grid_idx = slice_grid[:, None]
            x_selected = x[grid_idx, time_idx, :]

            if c is not None:
                c_select = c[slice_grid, :]

            x_selected = torch.from_numpy(x_selected).float()
            if c is not None:
                c_select = torch.from_numpy(c_select).float()

            return x_selected, c_select


        num_grid, num_time, _ = train_x_norm.shape
        nx = train_x_norm.shape[-1] + static_data_norm.shape[-1]
        seq_len = self.configs.seq_len
        train_warmup = self.configs.seq_len
        train_x_norm = train_x_norm

        bs = min(self.configs.batch_size, num_grid)
        total_iterations_per_epoch = int(
            np.ceil(np.log(0.01) / np.log(1 - bs * self.configs.seq_len / num_grid / (num_time - seq_len))))

        for epoch in np.arange(self.start_epoch, self.args.epochs):
            epoch += 1
            total_loss = 0
            for iteration in range(total_iterations_per_epoch):
                self.model.zero_grad()

                slice_grid, slice_time = random_index(num_grid, num_time, [bs, seq_len], train_warmup=train_warmup)
                sub_x, sub_c = select_subset(train_x_norm, slice_grid, slice_time, seq_len, c=static_data_norm,
                                      train_warmup=train_warmup)

                batch_data_dict = {"batch_x": sub_x, "batch_c": sub_c, "batch_time_series_mask_index": torch.empty(0),
                                   "batch_static_mask_index": torch.empty(0), "mode": "train", "batch_target_std": None}
                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }

                output_dict = self.model(batch_data_dict)
                output_dict["outputs_time_series"] = output_dict["outputs_time_series"][:,train_warmup:,:]
                output_dict["masked_time_series_index"] = output_dict["masked_time_series_index"][:,train_warmup:,:]
                output_dict["masked_missing_time_series_index"] = output_dict["masked_missing_time_series_index"][:,train_warmup:,:]


                loss = self.compute_loss(batch_data_dict, output_dict)

                # skip nan loss
                if torch.isnan(loss):
                    continue

                loss.backward()

                # clip gradients
                if self.args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

                self.model_optim.step()

                total_loss = total_loss + loss.item()

            total_loss = total_loss / total_iterations_per_epoch
            print("Epoch: %d, Loss: %.3f" % (epoch, total_loss))

            self.save_model(self.checkpoints_dir, epoch, self.model, self.model_optim, self.epoch_train_loss_list,
                            self.epoch_val_loss_list)

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)

            # # update the dataset index
            # train_loader.dataset.update_samples_index(True)

            self.epoch_train_loss_list.append(np.average(total_loss))
            self.plot_loss_curve(self.epoch_train_loss_list, None,
                                 os.path.join(self.pics_dir, "loss_curve_only_train.png"))

            if self.args.do_eval:
                # validation
                val_loss = self.vali(vali_loader)
                self.epoch_val_loss_list.append(val_loss)
                # plot the loss curve
                self.plot_loss_curve(self.epoch_train_loss_list, self.epoch_val_loss_list,
                                     os.path.join(self.pics_dir, "loss_curve.png"))

        print("results saved in: ", os.path.abspath(self.results_dir))

        return self.model

    def vali(self, vali_loader):

        # load the data
        train_x_norm = self.vali_data.raw_x
        static_data_norm = self.vali_data.raw_c

        def random_index(num_grid, num_time, dimSubset, train_warmup=0):
            bs, seq_len = dimSubset
            slice_grid = np.random.randint(0, num_grid, [bs])
            slice_time = np.random.randint(train_warmup, num_time - seq_len, [bs])
            return slice_grid, slice_time

        def select_subset(x, slice_grid, slice_time, seq_len, c=None, train_warmup=0):
            num_time = x.shape[1]
            if x.shape[0] == len(slice_grid):  # hack
                slice_grid = np.arange(0, len(slice_grid))  # hack
            if num_time <= seq_len:
                slice_time.fill(0)

            time_idx = slice_time[:, None] + np.arange(-train_warmup, seq_len)[None, :]
            grid_idx = slice_grid[:, None]
            x_selected = x[grid_idx, time_idx, :]

            if c is not None:
                c_select = c[slice_grid, :]

            x_selected = torch.from_numpy(x_selected).float()
            if c is not None:
                c_select = torch.from_numpy(c_select).float()

            return x_selected, c_select

        num_grid, num_time, _ = train_x_norm.shape
        nx = train_x_norm.shape[-1] + static_data_norm.shape[-1]
        seq_len = self.configs.seq_len
        train_warmup = self.configs.seq_len
        train_x_norm = train_x_norm

        bs = min(self.configs.batch_size, num_grid)
        total_iterations_per_epoch = int(
            np.ceil(np.log(0.01) / np.log(1 - bs * self.configs.seq_len / num_grid / (num_time - seq_len))))



        total_loss = []
        self.model.eval()
        with torch.no_grad():

            for iteration in range(total_iterations_per_epoch):
                self.model.zero_grad()

                slice_grid, slice_time = random_index(num_grid, num_time, [bs, seq_len], train_warmup=train_warmup)
                sub_x, sub_c = select_subset(train_x_norm, slice_grid, slice_time, seq_len, c=static_data_norm,
                                             train_warmup=train_warmup)

                batch_data_dict = {"batch_x": sub_x, "batch_c": sub_c, "batch_time_series_mask_index": torch.empty(0),
                                   "batch_static_mask_index": torch.empty(0), "mode": "val", "batch_target_std": None}
                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }
                batch_data_dict['mode'] = 'val'

                output_dict = self.model(batch_data_dict)
                output_dict["outputs_time_series"] = output_dict["outputs_time_series"][:, train_warmup:, :]
                output_dict["masked_time_series_index"] = output_dict["masked_time_series_index"][:, train_warmup:, :]
                output_dict["masked_missing_time_series_index"] = output_dict["masked_missing_time_series_index"][:,
                                                                  train_warmup:, :]

                loss = self.compute_loss(batch_data_dict, output_dict)

                # skip nan loss
                if torch.isnan(loss):
                    continue

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test=0):
        test_data, test_loader = self.test_data, self.test_loader

        if self.configs.do_test:
            print('loading model')
            self.load_model_resume()

        # load the data
        train_x_norm = self.train_data.raw_x
        static_data_norm = self.test_data.raw_c
        test_x_norm = self.test_data.raw_x

        test_time_len = test_x_norm.shape[1]
        # TestBuff = int(train_x_norm.shape[1] / 5) - 0  # Testing period
        # xTestBuff = train_x_norm[:test_x_norm.shape[0], -TestBuff:, :]
        # test_x_norm = np.concatenate([xTestBuff, test_x_norm], axis=1)

        num_grid, num_time, _ = test_x_norm.shape
        nx = train_x_norm.shape[-1] + static_data_norm.shape[-1]
        testBatch = 10
        index_grid_start = np.arange(0, num_grid, testBatch)
        index_grid_end = np.append(index_grid_start[1:], num_grid)

        preds, trues, preds_static, trues_static = [], [], [], []
        masked_index_time_series_list, masked_index_static_list = [], []
        self.model.eval()
        with torch.no_grad():
            for start, end in zip(index_grid_start, index_grid_end):
                sub_x = test_x_norm[start:end]
                sub_c = static_data_norm[start:end]

                sub_x = torch.from_numpy(sub_x).float()
                sub_c = torch.from_numpy(sub_c).float()

                batch_data_dict = {"batch_x": sub_x, "batch_c": sub_c, "batch_time_series_mask_index": torch.empty(0),
                                   "batch_static_mask_index": torch.empty(0), "mode": "test", "batch_target_std": None}
                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }
                batch_data_dict['mode'] = 'test'

                output_dict = self.model(batch_data_dict)

                outputs_time_series = output_dict["outputs_time_series"]  # [batch_size, pred_len, num_features]
                outputs_static = output_dict["outputs_static"]  # [batch_size, num_features]
                masked_time_series_index = output_dict["masked_time_series_index"]  # [batch_size, pred_len]
                masked_static_index = output_dict["masked_static_index"]  # [batch_size, num_features]

                pred = outputs_time_series[:, -test_time_len:, :]

                preds.append(pred.detach().cpu().numpy())
                preds_static.append(outputs_static.detach().cpu().numpy()[:, None, :])

                masked_index_time_series_list.append(masked_time_series_index.detach().cpu().numpy()[:, -test_time_len:, :])
                masked_index_static_list.append(masked_static_index.detach().cpu().numpy()[:, None, :])

        preds = np.concatenate(preds, axis=0)
        preds_static = np.concatenate(preds_static, axis=0)
        masked_index_time_series_list = np.concatenate(masked_index_time_series_list, axis=0)
        masked_index_static_list = np.concatenate(masked_index_static_list, axis=0)

        preds_restored = preds
        preds_static_restored = preds_static
        masked_index_time_series_list_restored = masked_index_time_series_list
        masked_index_static_list_restored = masked_index_static_list
        trues_restored = test_x_norm[:, -test_time_len:, :]
        trues_static_restored = static_data_norm[:,None,:]

        # # rescale the data
        # preds_rescale = test_data.inverse_transform(preds_restored, mean=test_data.scaler["x_mean"],
        #                                             std=test_data.scaler["x_std"])
        # trues_rescale = test_data.inverse_transform(trues_restored, mean=test_data.scaler["x_mean"],
        #                                             std=test_data.scaler["x_std"])
        #
        # preds_static_rescale = test_data.inverse_transform(preds_static_restored, mean=test_data.scaler["c_mean"],
        #                                                    std=test_data.scaler["c_std"], inverse_categorical=True)
        # trues_static_rescale = test_data.inverse_transform(trues_static_restored, mean=test_data.scaler["c_mean"],
        #                                                    std=test_data.scaler["c_std"], inverse_categorical=True)

        preds_rescale = preds
        trues_rescale = trues
        preds_static_rescale = preds_static
        trues_static_rescale = trues_static

        preds_static_rescale = preds_static_rescale[:, self.configs.static_pred_start_point, :]
        trues_static_rescale = trues_static_rescale[:, self.configs.static_pred_start_point, :]
        masked_index_static_list_restored = masked_index_static_list_restored[:, self.configs.static_pred_start_point,:]

        unmasked_index_time_series_list_restored = np.logical_not(masked_index_time_series_list_restored)
        unmasked_index_static_list_restored = np.logical_not(masked_index_static_list_restored)

        # replace the masked value with the np.nan to calculate the metrics
        preds_rescale[unmasked_index_time_series_list_restored] = np.nan
        trues_rescale[unmasked_index_time_series_list_restored] = np.nan

        preds_static_rescale[unmasked_index_static_list_restored] = np.nan
        trues_static_rescale[unmasked_index_static_list_restored] = np.nan

        # plot the final forecasting
        for idx_feature in range(preds_rescale.shape[-1]):
            visual(trues_rescale[0, :1000, idx_feature], preds_rescale[0, :1000, idx_feature],
                   os.path.join(self.pics_dir,
                                'feature_{}.pdf'.format(self.configs.time_series_variables[idx_feature])))

        time_series_median_metrics_dict = {key: [] for key in ["NSE", "KGE", "Corr"]}
        static_median_metrics_dict = {key: [] for key in ["NSE", "KGE", "Corr"]}
        f = open(os.path.join(self.args.output_dir, self.saved_folder, "results.txt"), 'a')
        # write the date and time
        f.write("Test date: {}, time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d"),
                                                 datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('\n')
        f.write(self.config_format(self.configs))
        f.write('\n')

        # calculate the metrics for each feature
        for idx_feature in range(preds_rescale.shape[-1]):
            metrics_list = ["NSE", "KGE", "Corr"]
            metrics_dict = cal_stations_metrics(trues_rescale[:, :, idx_feature], preds_rescale[:, :, idx_feature],
                                                metrics_list)
            print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.configs.time_series_variables[idx_feature],
                                                                      np.nanmedian(metrics_dict["NSE"]),
                                                                      np.nanmedian(metrics_dict["KGE"]),
                                                                      np.nanmedian(metrics_dict["Corr"])))
            f.write("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.configs.time_series_variables[idx_feature],
                                                                        np.nanmedian(metrics_dict["NSE"]),
                                                                        np.nanmedian(metrics_dict["KGE"]),
                                                                        np.nanmedian(metrics_dict["Corr"])))
            f.write('\n')
            time_series_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
            time_series_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
            time_series_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))
        f.write('\n')

        print("obs_mean: {:.3f}, obs_median: {:.7f}".format(np.nanmean(trues_rescale), np.nanmedian(trues_rescale)))
        print("pred_mean: {:.3f}, pred_median: {:.7f}".format(np.nanmean(preds_rescale), np.nanmedian(preds_rescale)))

        # for idx_feature_static in range(preds_static_rescale.shape[-1]):
        for idx_feature_static, static_variable in enumerate(self.configs.static_variables):
            if not static_variable in self.configs.static_variables_category:
                metrics_list = ["NSE", "KGE", "Corr"]
                metrics_dict = cal_stations_metrics(trues_static_rescale[:, idx_feature_static][None, :],
                                                    preds_static_rescale[:, idx_feature_static][None, :],
                                                    metrics_list, remove_neg=False)
                print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                    self.configs.static_variables[idx_feature_static],
                    np.nanmedian(metrics_dict["NSE"]),
                    np.nanmedian(metrics_dict["KGE"]),
                    np.nanmedian(metrics_dict["Corr"])))
                static_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                static_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                static_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

                f.write(
                    "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                        self.configs.static_variables[idx_feature_static],
                        np.nanmedian(metrics_dict["NSE"]),
                        np.nanmedian(metrics_dict["KGE"]),
                        np.nanmedian(metrics_dict["Corr"])))
                f.write('\n')
            else:
                metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
                sub_pred = preds_static_rescale[:, idx_feature_static]
                sub_true = trues_static_rescale[:, idx_feature_static]

                # remove the nan value
                index_nan = np.isnan(sub_pred) | np.isnan(sub_true)
                sub_pred = sub_pred[~index_nan]
                sub_true = sub_true[~index_nan]

                Accuracy = accuracy_score(sub_true, sub_pred)
                Precision = precision_score(sub_true, sub_pred, average='micro')
                Recall = recall_score(sub_true, sub_pred, average='micro')
                F1 = f1_score(sub_true, sub_pred, average='micro')
                print("{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                    self.configs.static_variables[idx_feature_static],
                    Accuracy,
                    Precision,
                    Recall,
                    F1))
                static_median_metrics_dict["NSE"].append(0)
                static_median_metrics_dict["KGE"].append(0)
                static_median_metrics_dict["Corr"].append(0)

                f.write(
                    "{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                        self.configs.static_variables[idx_feature_static],
                        Accuracy,
                        Precision,
                        Recall,
                        F1))
                f.write('\n')

        f.write('\n')
        f.close()

        # np.save(self.results_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.results_dir, 'pred.npy'), preds_rescale)
        np.save(os.path.join(self.results_dir, 'true.npy'), trues_rescale)

        # plot the metrics
        for key in ["NSE", "KGE", "Corr"]:
            saved_file = os.path.join(self.pics_dir, "metrics_{}.png".format(key))
            self.plot_time_series_statics_metrics_bar(saved_file,
                                                      key,
                                                      time_series_median_metrics_dict[key],
                                                      self.configs.time_series_variables,
                                                      static_median_metrics_dict[key],
                                                      self.configs.static_variables)

        return

    def inference(self):
        pass

    @staticmethod
    def plot_time_series_statics_metrics_bar(saved_file, metric_name,
                                             metric_time_series=None,
                                             time_series_variables=None,
                                             metric_static=None,
                                             static_variables=None):
        """
        Plot two types of metrics: time series and statics, in a 2-row subplot.

        metric_time_series: An array of time series metrics [features]
        metric_static: An array of static metrics [features]
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
        font_size = 14

        # Plotting horizontal bar chart for the time series metrics
        if metric_time_series is not None:
            metric_time_series = np.array(metric_time_series)
            # convert inf, -inf negative and nan to 0
            index = np.isinf(metric_time_series) | np.isnan(metric_time_series) | (metric_time_series < 0)
            metric_time_series[index] = 0

            ax1.barh(np.arange(len(metric_time_series)), metric_time_series, color='skyblue')
            # ax1.set_xlabel(metric_name)

        # set time series variables as yticks
        if time_series_variables is not None:
            ax1.set_yticks(np.arange(len(metric_time_series)))
            ax1.set_yticklabels(time_series_variables, fontsize=font_size)
            ax1.tick_params(axis='y', rotation=45)
            # # set xlim
            ax1.set_xlim([0, 1])

        # set xticks font size
        ax1.tick_params(axis='x', labelsize=font_size)

        # Plotting vertical bar chart for the static metrics
        if metric_static is not None:
            metric_static = np.array(metric_static)
            # convert inf, -inf negative and nan to 0
            index = np.isinf(metric_static) | np.isnan(metric_static) | (metric_static < 0)
            metric_static[index] = 0
            ax2.bar(np.arange(len(metric_static)), metric_static, color='salmon')
            # ax2.set_ylabel(metric_name)

            ax2.set_ylim([0, 1])

        # set static variables as xticks
        if static_variables is not None:
            ax2.set_xticks(np.arange(len(metric_static)))
            ax2.set_xticklabels(static_variables, fontsize=font_size)
            ax2.tick_params(axis='x', rotation=60)

            for label in ax2.get_xticklabels():
                label.set_horizontalalignment('right')

        # set yticks font size
        ax2.tick_params(axis='y', labelsize=font_size)

        # set title for the whole figure
        fig.suptitle(metric_name, fontsize=font_size + 2)

        # Adjusting layout and displaying the plot
        plt.tight_layout()
        # save
        plt.savefig(saved_file)
