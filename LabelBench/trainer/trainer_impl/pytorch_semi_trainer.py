import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LabelBench.trainer.trainer_impl.pytorch_passive_trainer import PyTorchPassiveTrainer
from LabelBench.dataset.feature_extractor import make_semi_transforms
from LabelBench.trainer.utils import effective_max_epoch, returned_model_step, should_validate


class PyTorchSemiTrainer(PyTorchPassiveTrainer):
    trainer_name = None

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, feature_extractor):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn,
                         model_config, metric, feature_extractor)
        self.use_strong = None

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us, iter):
        raise NotImplementedError(
            "Subclass does not have implementation of semi-supervised learning training function.")

    def initialize_trainer(self):
        pass

    def train(self, finetune_model=None, finetune_config=None):
        if self.use_strong is None:
            raise NotImplementedError("Subclass does not specify strong transformation use in use_strong.")

        model, params, loss_fn, max_epoch, optimizer, scheduler, early_stopping = self.init_train(finetune_model)
        min_steps = self.trainer_config.get("min_steps")
        max_epoch = effective_max_epoch(max_epoch, len(self.dataset.labeled_idxs()),
                                        self.trainer_config["train_batch_size"], min_steps,
                                        self.trainer_config.get("max_steps"))
        self.num_steps_taken = 0
        self.ended_by = "epoch_limit"
        self.best_step = None
        last_validation_step = None
        mixup_fn = self.trainer_config["mixup_fn"](self.dataset.num_classes)

        # Get the training dataset for the non-embedding dataset.
        if "use_embeddings" not in self.trainer_config or (not self.trainer_config["use_embeddings"]):
            train_dataset, _, _ = self.dataset.get_input_datasets()
            if "use_customized_transform" in self.model_config and self.model_config["use_customized_transform"]:
                transform = model.module.get_preprocess(split="train")
            else:
                transform = train_dataset.get_transform()

            transform_weak, transform_strong = make_semi_transforms(transform)
            train_dataset.set_transform(transform_weak)
            if self.use_strong:
                assert transform_strong is not None, "Strong transform cannot be None"
                train_dataset.set_strong_transform(transform_strong)

        # Executing initialization before training (if implemented).
        self.initialize_trainer()

        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
            self.stopped_epoch = epoch + 1

            # For each epoch, update the embedding dataset and use the saved embedding dataset epoch.
            if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
                self.dataset.update_embedding_dataset(epoch=epoch, feature_extractor=self.feature_extractor,
                                                      use_strong=self.use_strong)
                train_dataset, _, _ = self.dataset.get_embedding_datasets()

            class_weights = 1. / np.clip(np.sum(self.dataset.get_train_labels(), axis=0), a_min=1, a_max=None)
            class_weights = torch.from_numpy(class_weights).float().cuda()

            # Set dataset to return indices.
            train_dataset.set_return_indices(True)

            if epoch == 0 or ("use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]):
                # Make labeled loader.
                labeled_dataset = Subset(train_dataset, self.dataset.labeled_idxs())
                labeled_loader = DataLoader(labeled_dataset,
                                            batch_size=self.trainer_config["train_batch_size"],
                                            shuffle=True,
                                            num_workers=self.trainer_config["num_workers"],
                                            drop_last=(len(labeled_dataset) >= self.trainer_config["train_batch_size"]))

                # Make unlabeled loader.
                unlabeled_dataset = Subset(train_dataset, self.dataset.unlabeled_idxs())
                unlabeled_loader = \
                    DataLoader(unlabeled_dataset,
                               batch_size=self.trainer_config["uratio"] * self.trainer_config["train_batch_size"],
                               shuffle=True,
                               num_workers=self.trainer_config["num_workers"],
                               drop_last=(len(unlabeled_dataset) >= self.trainer_config["train_batch_size"]))
            unlabeled_iterator = iter(unlabeled_loader)

            for img_l, target_l, *other_l in tqdm(labeled_loader, desc="Batch Index"):
                try:
                    img_u, _, idx_u = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    img_u, _, idx_u = next(unlabeled_iterator)

                if self.use_strong:
                    img_uw, img_us = img_u
                    img_us = img_us.float()
                    img_l = img_l[0]
                else:
                    img_uw = img_u
                    img_us = None

                img_l, target_l = img_l.float(), target_l.float()
                img_uw = img_uw.float()

                if mixup_fn is not None:
                    # Since the target is one-hot and the mixup function accepts class index only, we need to convert it
                    # to the class index.
                    target_l = torch.argmax(target_l, dim=1)
                    img, target_l = mixup_fn(img_l, target_l)

                with torch.cuda.amp.autocast():
                    loss = self.train_step(model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us, counter)

                counter = self.scheduler_step(scheduler, counter)
                optimizer.zero_grad()
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()
                self.num_steps_taken += 1

            train_dataset.set_return_indices(False)

            if should_validate(self.num_steps_taken, last_validation_step, min_steps,
                               self.trainer_config.get("val_every_steps"), epoch == max_epoch - 1):
                last_validation_step = self.num_steps_taken
                if self.check_early_stop(early_stopping, model):
                    self.ended_by = "early_stopping"
                    break
                self.best_step = returned_model_step(self.best_step, self.num_steps_taken, early_stopping)

        # Set embeddings back to not using semi-supervised transformation for evaluation.
        if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
            self.dataset.update_embedding_dataset(epoch=0, feature_extractor=self.feature_extractor, use_strong=False)
        elif self.use_strong:
            train_dataset.set_strong_transform(None)

        self.best_val_loss = None
        if early_stopping is not None:
            model.load_state_dict(early_stopping.best_state_dict)
            self.best_val_loss = float(early_stopping.val_loss_min)
        return model
