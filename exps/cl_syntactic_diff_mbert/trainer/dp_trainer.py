import math
import copy
import time
from datetime import datetime
import random
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Union, cast
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.sampler import Sampler as TorchSampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

from calf.utils.distributed import is_dist

from calf import device, OUTPUT, logger, CHECKPOINT, LOG
from calf.utils.log import log_line
from calf.utils.optimizer import LinearSchedulerWithWarmup, AnnealOnPlateau

from calf.utils.corpus import Corpus, UDTreebankCorpus
from calf.utils.dataset import Dataset, build_dataloader
from calf.utils.transform import Transform

from calf.utils import enumeration as enum


class DPTrainer(object):
    def __init__(self, cfg: DictConfig, transform: Transform, model: nn.Module):
        self.transform = transform
        self.model = model
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.checkpoint_path = None
        self.log_path = None

        self.train_log_file = self.trainer_cfg.get("train_log_file", "train.log")
        self.log_loss = self.trainer_cfg.get("log_loss", True)
        self.loss_log_file = self.trainer_cfg.get("loss_log_file", "loss.log")
        self.log_k_times = self.trainer_cfg.get("log_k_times", 10)
        self.save_model_each_k_epochs = self.trainer_cfg.get("save_model_each_k_epochs", 0)
        self.checkpoint = self.trainer_cfg.get("checkpoint", False)
        self.save_final_model = self.trainer_cfg.get("save_final_model", True)

        self.mini_batch_size = self.trainer_cfg.get("mini_batch_size", 8)
        self.eval_batch_size = self.trainer_cfg.get("eval_batch_size", None)

        self.min_learning_rate = self.trainer_cfg.get("min_learning_rate", 1.0e-6)

        self.max_epochs = self.trainer_cfg.get("max_epochs", 5)

        self.train_with_dev = self.trainer_cfg.get("train_with_dev", False)
        self.shuffle = self.trainer_cfg.get("shuffle", False)

        self.eval_on_train_fraction = self.trainer_cfg.get("eval_on_train_fraction", "dev")
        self.eval_on_train_shuffle = self.trainer_cfg.get("eval_on_train_shuffle", False)
        self.main_evaluation_metric = self.trainer_cfg.get("main_evaluation_metric", "accuracy")
        self.use_final_model_for_eval = self.trainer_cfg.get("use_final_model_for_eval", False)

        self.anneal_with_restarts = self.trainer_cfg.get("anneal_with_restarts", False)
        self.anneal_with_prestarts = self.trainer_cfg.get("anneal_with_prestarts", False)
        self.anneal_against_dev_loss = self.trainer_cfg.get("anneal_against_dev_loss", False)

        self.num_workers = self.trainer_cfg.get("num_workers", None)

        self.optimizer_cfg = self.trainer_cfg.optimizer
        self.scheduler_cfg = self.trainer_cfg.scheduler

    def train(
            self,
            dataset_cfg: Dict,
            epoch: int = 0,
    ) -> Dict:
        """
        Trains any class that implements the elephant.models.MotelTemplate interface.
        """
        # ----------------------------------------------------------------------------------------------------
        # Initialize dataset and output directory
        # ----------------------------------------------------------------------------------------------------
        train_corpus = UDTreebankCorpus(lang=dataset_cfg["lang"], genre=dataset_cfg["genre"], split=enum.Split.TRAIN)
        dev_corpus = UDTreebankCorpus(lang=dataset_cfg["lang"], genre=dataset_cfg["genre"], split=enum.Split.DEV)
        test_corpus = UDTreebankCorpus(lang=dataset_cfg["lang"], genre=dataset_cfg["genre"], split=enum.Split.TEST)
        # transform = UDTreebankTransform(tokenizer=self.model.encoder.tokenizer)
        self.train_dataset = Dataset(corpus=train_corpus, transform=self.transform, cache=True,
                                     max_len=self.cfg.transform.max_len)
        self.dev_dataset = Dataset(corpus=dev_corpus, transform=self.transform, cache=True,
                                   max_len=self.cfg.transform.max_len)
        self.test_dataset = Dataset(corpus=test_corpus, transform=self.transform, cache=True,
                                    max_len=self.cfg.transform.max_len)

        lang = dataset_cfg["lang"]
        self.checkpoint_path = CHECKPOINT / self.cfg.exp_name / lang
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        self.log_path = LOG / self.cfg.exp_name / lang
        self.log_path.mkdir(exist_ok=True, parents=True)

        self.train_log_file = self.log_path / Path(self.train_log_file).name
        self.loss_log_file = self.log_path / Path(self.loss_log_file).name

        result_dict = {
            "val_uas": 0.0, "val_las": 0.0, "test_uas": 0.0, "test_las": 0.0,
        }

        # ----------------------------------------------------------------------------------------------------
        # Validate parameters and check preconditions
        # ----------------------------------------------------------------------------------------------------
        assert self.train_dataset, "Check training data."

        # ----------------------------------------------------------------------------------------------------
        # Prepare training environment and parameters
        # ----------------------------------------------------------------------------------------------------
        # check for and delete previous best models
        self._check_for_and_delete_previous_best_models()

        # prepare loss logging file and set up header
        if self.log_loss and self.loss_log_file:
            open(self.loss_log_file, mode='w', encoding="utf-8").close()
        else:
            self.loss_log_file = None

        # set train and evaluation batch size
        if self.eval_batch_size is None:
            self.eval_batch_size = self.mini_batch_size

        # ----------------------------------------------------------------------------------------------------
        # Build optimizer
        # ----------------------------------------------------------------------------------------------------
        optimizer = self._build_optimizer(self.optimizer_cfg)

        initial_learning_rate = [group["lr"] for group in optimizer.param_groups]

        min_learning_rate = [self.min_learning_rate] * len(initial_learning_rate)
        for i, lr in enumerate(initial_learning_rate):
            if lr < min_learning_rate[i]:
                min_learning_rate[i] = lr / 10

        optimizer = cast(torch.optim.Optimizer, optimizer)

        # ----------------------------------------------------------------------------------------------------
        # Build scheduler
        # ----------------------------------------------------------------------------------------------------
        anneal_mode = "min" if self.train_with_dev or self.anneal_against_dev_loss else "max"
        best_validation_score = math.inf if self.train_with_dev or self.anneal_against_dev_loss else -1.0

        dataset_size = len(self.train_dataset)  # use source dataset size
        if self.train_with_dev:
            dataset_size += len(self.dev_dataset)

        if self.scheduler_cfg.name == "OneCycleLR":
            steps_per_epoch = int((dataset_size + self.mini_batch_size - 1) / self.mini_batch_size)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=initial_learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=self.max_epochs - epoch,  # if we load a checkpoint, we have already trained for epoch
                pct_start=0.0,
                cycle_momentum=self.scheduler_cfg.cycle_momentum,
            )
        elif self.scheduler_cfg.name == "LinearSchedulerWithWarmup":
            steps_per_epoch = int((dataset_size + self.mini_batch_size - 1) / self.mini_batch_size)
            num_train_steps = steps_per_epoch * self.max_epochs
            num_warmup_steps = num_train_steps * self.scheduler_cfg.warmup_fraction \
                if self.scheduler_cfg.warmup_fraction > 0 else 1
            scheduler = LinearSchedulerWithWarmup(
                optimizer,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
            )
        elif self.scheduler_cfg.name == "AnnealOnPlateau":
            scheduler = AnnealOnPlateau(
                optimizer,
                mode=anneal_mode,
                factor=self.scheduler_cfg.anneal_factor,
                patience=self.scheduler_cfg.patience,
                verbose=True,
                initial_extra_patience=self.scheduler_cfg.initial_extra_patience,
            )
        else:
            raise NotImplementedError(f"Scheduler {self.scheduler_cfg.name} not implemented.")

        log_bad_epochs = True if isinstance(scheduler, AnnealOnPlateau) else False

        # ----------------------------------------------------------------------------------------------------
        # Prepare training data
        # ----------------------------------------------------------------------------------------------------
        s_train_data = self.train_dataset

        if self.train_with_dev:
            s_parts = [self.train_dataset]
            if self.train_with_dev and self.dev_dataset:
                s_parts.append(self.dev_dataset)
            s_train_data = torch.utils.data.ConcatDataset(s_parts)

        # if self.trainer_cfg.subsample_train_data:
        #     train_indices = list(range(len(self.train_dataset)))
        #     if self.trainer_cfg.shuffle_subsample:
        #         train_indices = list(range(len(self.train_dataset)))
        #         random.shuffle(train_indices)
        #         subsample_indices = train_indices[:self.trainer_cfg.subsample_t_train_data_num]
        #     else:
        #         subsample_indices = list(range(self.trainer_cfg.subsample_t_train_data_num))
        #     s_train_data = torch.utils.data.Subset(s_train_data, subsample_indices)

        # ----------------------------------------------------------------------------------------------------
        # Prepare evaluation data setting
        # ----------------------------------------------------------------------------------------------------
        # whether using dev dataset to evaluate
        if not self.train_with_dev and self.dev_dataset:
            log_dev = True
        else:
            log_dev = False

        # whether using part of train dataset to evaluate
        if self.eval_on_train_fraction == "dev" or self.eval_on_train_fraction > 0.0:
            log_train_part = True
        else:
            log_train_part = False

        # prepare certain amount of training data for evaluation
        if log_train_part:
            if self.eval_on_train_fraction == "dev":
                s_train_part_size = len(self.dev_dataset)
            else:
                s_train_part_size = int(len(self.train_dataset) * self.eval_on_train_fraction)

            assert s_train_part_size > 0

            if not self.eval_on_train_shuffle:
                s_train_part_indices = list(range(s_train_part_size))
                s_train_part = torch.utils.data.Subset(self.train_dataset, s_train_part_indices)

        # ----------------------------------------------------------------------------------------------------
        # Build data sampler
        # ----------------------------------------------------------------------------------------------------
        s_sampler = self._build_sampler(s_train_data)

        # ----------------------------------------------------------------------------------------------------
        # Start training
        # ----------------------------------------------------------------------------------------------------
        if epoch >= self.max_epochs:
            logger.warning(
                f"Starting at epoch {epoch + 1}/{self.max_epochs}. No training will be done."
            )

        train_seqids = []

        train_loss_history = []

        dev_loss_history = []
        dev_score_history = []

        # at any point you can hit Ctrl + C to break out of training early.
        try:
            lr_info = ",".join([f"{lr:.8f}" for lr in initial_learning_rate])

            log_line(logger)
            logger.info(f'Model: "{self.model.model_cfg.name}"')
            log_line(logger)
            logger.info("Parameters:")
            logger.info(f' - learning_rate: "{lr_info}"')
            logger.info(f' - mini_batch_size: "{self.mini_batch_size}"')
            logger.info(f' - patience: "{self.scheduler_cfg.patience}"')
            logger.info(f' - anneal_factor: "{self.scheduler_cfg.anneal_factor}"')
            logger.info(f' - max_epochs: "{self.max_epochs}"')
            logger.info(f' - shuffle: "{self.shuffle}"')
            logger.info(f' - train_with_dev: "{self.train_with_dev}"')
            log_line(logger)
            logger.info(f'Model training base path: "{OUTPUT}"')
            logger.info(f'- checkpoint path: "{self.checkpoint_path}"')
            logger.info(f'- log path: "{self.log_path}"')
            log_line(logger)
            logger.info(f"Device: {device}")

            # ------------------------------------------------------------------------------------------------
            # Loop epochs
            # ------------------------------------------------------------------------------------------------
            previous_learning_rate = initial_learning_rate
            momentum = [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

            self.set_requires_grad(self.model.encoder, requires_grad=self.trainer_cfg.train_encoder)
            self.set_requires_grad(self.model.parser, requires_grad=True)

            for epoch in range(epoch + 1, self.max_epochs + 1):
                log_line(logger)

                # --------------------------------------------------------------------------------------------
                # Prepare variables before loop batches
                # --------------------------------------------------------------------------------------------
                # shuffle train part data for evaluation

                if self.eval_on_train_shuffle:
                    s_train_part_indices = list(range(len(self.train_dataset)))
                    random.shuffle(s_train_part_indices)
                    s_train_part_indices = s_train_part_indices[:s_train_part_size]  # noqa
                    s_train_part = torch.utils.data.Subset(self.train_dataset, s_train_part_indices)

                # reset current learning rate
                current_learning_rate = [group["lr"] for group in optimizer.param_groups]

                # update batch size if learning rate changed
                lr_changed = any(
                    [lr != prev_lr for lr, prev_lr in zip(current_learning_rate, previous_learning_rate)]
                )

                # stop training if learning rate becomes too small
                all_lrs_too_small = all(
                    [lr < min_lr for lr, min_lr in zip(current_learning_rate, min_learning_rate)]
                )
                if not isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)) and all_lrs_too_small:
                    log_line(logger)
                    logger.info("Learning rate too small, quitting training!")
                    log_line(logger)
                    break

                # reload last best model if annealing with restarts is enabled
                if (
                        (self.anneal_with_restarts or self.anneal_with_prestarts)
                        and lr_changed
                        and (self.checkpoint_path / "best_model.pt").is_file()
                ):
                    if self.anneal_with_restarts:
                        logger.info("Resetting to best model.")
                        self.model.load_state_dict(
                            self.model.load(self.checkpoint_path / "best_model.pt").state_dict()
                        )
                    if self.anneal_with_prestarts:
                        logger.info("Resetting to pre-best model.")
                        self.model.load_state_dict(
                            self.model.load(self.checkpoint_path / "pre_best_model.pt").state_dict()
                        )
                s_dataloader = DataLoader(
                    s_train_data,
                    batch_size=self.mini_batch_size,
                    shuffle=self.shuffle,  # if epoch > 1 else False
                    sampler=s_sampler,
                    num_workers=0 if self.num_workers is None else self.num_workers,
                    collate_fn=self.transform.collate,
                    drop_last=False
                )

                # --------------------------------------------------------------------------------------------
                # Reserve model states before running batches
                # --------------------------------------------------------------------------------------------
                if self.anneal_with_prestarts:
                    last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

                previous_learning_rate = current_learning_rate

                # --------------------------------------------------------------------------------------------
                # Start a training epoch
                # --------------------------------------------------------------------------------------------
                self.model.train()

                train_loss: float = 0
                seen_batches = 0
                total_number_of_batches = len(s_dataloader)
                modulo = max(1, int(total_number_of_batches / self.log_k_times))

                batch_time = 0.0
                average_over = 0

                s_iter = iter(s_dataloader)
                num_iterator = len(s_dataloader)

                for batch_no in range(1, num_iterator):
                    s_batch = next(s_iter)
                    start_time = time.time()

                    self.model.zero_grad(set_to_none=True)
                    optimizer.zero_grad()

                    # forward loss
                    loss_result, count = self.model.forward_loss(batch=s_batch)

                    loss = loss_result.metric_score
                    loss.backward()

                    average_over += count
                    train_loss += loss.item() * count

                    # clip gradient norm
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                    # optimizer step
                    optimizer.step()

                    # do the scheduler step if one-cycle or linear decay
                    if isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)):
                        scheduler.step()
                        current_learning_rate = [group["lr"] for group in optimizer.param_groups]
                        momentum = [
                            group["betas"][0] if "betas" in group else group.get("momentum", 0)
                            for group in optimizer.param_groups
                        ]

                    # log training info each modulo iterations
                    seen_batches += 1
                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        momentum_info = ""
                        if self.scheduler_cfg.cycle_momentum:
                            momentum_info = " - momentum:" + ",".join([f"{m:.4f}" for m in momentum])

                        lr_info = ",".join([f"{lr:.8f}" for lr in current_learning_rate])

                        if average_over > 0:
                            intermittent_loss = train_loss / average_over
                        else:
                            intermittent_loss = train_loss / seen_batches

                        logger.info(
                            f"epoch: {epoch}"
                            f" - iter: {seen_batches}/{total_number_of_batches}"
                            f" - loss: {intermittent_loss:.8f}"
                            f" - time (sec): {(time.time() - start_time):.2f}"
                            f" - samples/sec: {average_over / (time.time() - start_time):.2f}"
                            f" - lr: {lr_info}{momentum_info}"
                        )
                        logger.info(loss_result.log_header)
                        logger.info(loss_result.log_line)
                        batch_time = 0.0

                if average_over != 0:
                    train_loss /= average_over

                # --------------------------------------------------------------------------------------------
                # Start a training evaluation
                # --------------------------------------------------------------------------------------------
                self.model.eval()

                # save model each K epochs
                if self.save_model_each_k_epochs > 0 and epoch % self.save_model_each_k_epochs == 0:
                    logger.info("Saving model trained after current epoch.")
                    model_name = "model_epoch_" + str(epoch) + ".pt"
                    self.model.save(self.checkpoint_path / model_name)

                log_line(logger)
                logger.info(f"EPOCH {epoch} done: loss {train_loss:.4f} - lr {lr_info}.")

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train_part:
                    eval_dataloader = DataLoader(
                        dataset=s_train_part,  # noqa
                        batch_size=self.eval_batch_size,
                        num_workers=self.num_workers,  # noqa
                        collate_fn=self.transform.collate,
                        shuffle=False,
                        drop_last=False
                    )
                    s_train_part_metric_result, s_train_part_loss_result = self.model.evaluate(
                        dataloader=eval_dataloader
                    )
                    result_line += f"\t{s_train_part_metric_result.log_line}"
                    result_line += f"\t{s_train_part_loss_result.log_line}"

                    logger.info(f"TRAIN_PART")
                    logger.info(f"{s_train_part_metric_result.log_header}")
                    logger.info(f"{s_train_part_metric_result.log_line}")
                    logger.info(f"{s_train_part_loss_result.log_header}")
                    logger.info(f"{s_train_part_loss_result.log_line}")

                if log_dev:
                    assert self.dev_dataset
                    eval_dataloader = DataLoader(
                        dataset=self.dev_dataset,
                        batch_size=self.eval_batch_size,
                        num_workers=self.num_workers,  # noqa
                        collate_fn=self.transform.collate,
                        shuffle=False,
                        drop_last=False
                    )
                    s_dev_metric_result, s_dev_loss_result = self.model.evaluate(
                        dataloader=eval_dataloader
                    )
                    result_line += f"\t{s_dev_metric_result.log_line}"
                    result_line += f"\t{s_dev_loss_result.log_line}"

                    logger.info(f"DEV")
                    logger.info(f"{s_dev_metric_result.log_header}")
                    logger.info(f"{s_dev_metric_result.log_line}")
                    logger.info(f"{s_dev_loss_result.log_header}")
                    logger.info(f"{s_dev_loss_result.log_line}")

                    dev_score = s_dev_metric_result.metric_score
                    dev_loss = s_dev_loss_result.metric_score

                    dev_score_history.append(dev_score)
                    dev_loss_history.append(dev_loss)

                # determine if this is the best model or if we need to anneal
                current_epoch_has_best_model_so_far = False
                # default mode: anneal against dev score
                if log_dev and not self.anneal_against_dev_loss:
                    if dev_score > best_validation_score:  # noqa
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_score
                        result_dict["val_las"] = s_dev_metric_result.metric_detail["las"]
                        result_dict["val_uas"] = s_dev_metric_result.metric_detail["uas"]

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(dev_score, dev_loss)  # noqa

                # alternative: anneal against dev loss
                if log_dev and self.anneal_against_dev_loss:
                    if dev_loss < best_validation_score:  # noqa
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_loss

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(dev_loss)

                # alternative: anneal against train loss
                if self.train_with_dev:
                    if train_loss < best_validation_score:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = train_loss

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(train_loss)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except AttributeError:
                    bad_epochs = 0

                new_learning_rate = [group["lr"] for group in optimizer.param_groups]
                if any([
                    new_lr != prev_lr for new_lr, prev_lr in zip(new_learning_rate, previous_learning_rate)
                ]):
                    bad_epochs = self.scheduler_cfg.patience + 1

                    # lr unchanged
                    if all([
                        prev_lr == initial_lr
                        for prev_lr, initial_lr in zip(previous_learning_rate, initial_learning_rate)
                    ]):
                        bad_epochs += self.scheduler_cfg.initial_extra_patience

                # log bad epochs
                if log_bad_epochs:
                    logger.info(f"BAD EPOCHS (no improvement): {bad_epochs}")  # noqa

                if self.loss_log_file is not None:
                    with open(self.loss_log_file, mode='a') as f:
                        # make headers on first epoch
                        if epoch == 1:
                            bad_epoch_header = "BAD_EPOCHS\t" if log_bad_epochs else ""
                            f.write(f"EPOCH\tTIMESTAMP\t{bad_epoch_header}LEARNING_RATE\tTRAIN_LOSS")

                            if log_train_part:
                                f.write(
                                    "\tS_TRAIN_".join(s_train_part_metric_result.log_header.split("\t"))  # noqa
                                )
                                f.write(
                                    "\tS_TRAIN_".join(s_train_part_loss_result.log_header.split("\t"))  # noqa
                                )

                            if log_dev:
                                f.write(
                                    "\tS_DEV_".join(s_dev_metric_result.log_header.split("\t"))  # noqa
                                )
                                f.write(
                                    "\tS_DEV_".join(s_dev_loss_result.log_header.split("\t"))  # noqa
                                )

                        lr_info = ",".join([f"{lr:.8f}" for lr in current_learning_rate])
                        bad_epoch_info = "\t" + str(bad_epochs) if log_bad_epochs else ""  # noqa
                        f.write(
                            f"\n{epoch}\t{datetime.now():%H:%M:%S}{bad_epoch_info}\t{lr_info}\t{train_loss}"
                        )
                        f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if self.checkpoint:
                    self.model.save(self.checkpoint_path / "checkpoint.pt", checkpoint=True)

                # check whether to save best model
                if (
                        (not self.train_with_dev or self.anneal_with_restarts or self.anneal_with_prestarts)
                        and current_epoch_has_best_model_so_far  # noqa
                        and not self.use_final_model_for_eval
                ):
                    logger.info("Saving the best model.")
                    self.model.save(self.checkpoint_path / "best_model.pt")

                    if self.anneal_with_prestarts:
                        current_state_dict = self.model.state_dict()
                        self.model.load_state_dict(last_epoch_model_state_dict)  # noqa
                        self.model.save(self.checkpoint_path / "pre_best_model.pt")
                        self.model.load_state_dict(current_state_dict)

            # if not in parameter selection mode, save final model
            if self.save_final_model:
                self.model.save(self.checkpoint_path / "final_model.pt")

        except KeyboardInterrupt:
            log_line(logger)
            logger.info("Exiting from training early.")

            logger.info("Saving models ...")
            self.model.save(self.checkpoint_path / "final_model.pt")
            logger.info("Done.")

        finally:
            pass

        # test best models if test data is present
        if self.test_dataset is not None:
            final_score, test_metric_result = self.final_test(
                checkpoint_path=self.checkpoint_path
            )
            result_dict["test_las"] = test_metric_result.metric_detail["las"]
            result_dict["test_uas"] = test_metric_result.metric_detail["uas"]
        else:
            final_score = 0
            logger.info("Test data not provided, setting final score to 0.")

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
            "result_dict": result_dict,
            "train_seqids": train_seqids
        }

    def final_test(
            self,
            checkpoint_path: Union[Path, str],
    ):
        log_line(logger=logger)

        self.model.eval()

        if (checkpoint_path / "best_model.pt").exists():
            self.model.load_state_dict(self.model.load(checkpoint_path / "best_model.pt").state_dict())
        else:
            logger.info("Testing using last state of model ...")

        assert self.test_dataset
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,  # noqa
            collate_fn=self.transform.collate,
            shuffle=False,
            drop_last=False
        )
        s_test_metric_result, s_test_loss_result = self.model.evaluate(dataloader=test_dataloader)

        logger.info(f"TEST")
        logger.info(s_test_metric_result.log_header)
        logger.info(s_test_metric_result.log_line)
        log_line(logger=logger)
        final_score = s_test_metric_result.metric_score

        return final_score, s_test_metric_result

    def _check_for_and_delete_previous_best_models(self):
        best_models = [f for f in self.checkpoint_path.glob("best_model*") if f.is_file()]
        for model in best_models:
            model.unlink()

    def _build_optimizer(self, optimizer_cfg) -> torch.optim.Optimizer:
        if optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay
            )
        elif optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
                momentum=optimizer_cfg.momentum
            )
        else:
            raise NotImplementedError(f'Optimizer "{optimizer_cfg.name}" is not implemented.')

        return optimizer

    @staticmethod
    def _build_sampler(dataset: Dataset) -> Optional[TorchSampler]:
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        ) if is_dist() else None

        return sampler

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
