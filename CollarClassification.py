import os
import torch
import matplotlib.pyplot as plt
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class TestDatasetViT(Dataset):
    def __init__(self, data_path, imgs, feature_extractor=None):
        self.data_path = data_path
        self.imgs = imgs
        self.W = 224
        self.H = 224
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.imgs)

    # load image from disk
    def _load_train_image(self, fn):
        # мб добавить cv.resize() 
        img = cv.imread(filename=os.path.join(self.data_path, fn))
        img = cv.resize(img, (self.W, self.H), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.float32(img) / 255

        return img

    def _load_train_image_for_extractor(self, fn):
        img = cv.imread(filename=os.path.join(self.data_path, fn))
        img = cv.resize(img, (self.W, self.H), interpolation=cv.INTER_AREA)
        return img

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image = self._load_train_image_for_extractor(self.imgs[idx])
        inpt = self.feature_extractor(images=image, return_tensors="pt")
        channels = 3
        pixel_values = inpt['pixel_values'].view(3, self.feature_extractor.size, self.feature_extractor.size)
        image = self._load_train_image(self.imgs[idx])
        sample = image_name, torch.tensor(image).type(torch.float), pixel_values
        return sample
    
class CollarTagger(pl.LightningModule):
    def __init__(self, n_classes: int, ViT_model, n_training_steps=None, n_warmup_steps=None, n_hidden_layers_cat=12):
        super().__init__()
        self.n_hidden_layers_cat = n_hidden_layers_cat
        self.ViT_model = ViT_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.ViT_model.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, imgs, pixel_values, labels=None):
        output = self.ViT_model(pixel_values)
        last_hidden_state = output.last_hidden_state
        pooler_output = output.pooler_output # but we concat last hidden_layers
        #print('Pooler output: ', pooler_output.shape)
        hidden_states = output.hidden_states
        #print('Hidden_states ', len(hidden_states))
        summed_last_cat_layers = torch.stack(hidden_states[-self.n_hidden_layers_cat:]).sum(0)
        pooled_vector = torch.mean(summed_last_cat_layers, dim=1) # may be better than sum(), or we can use max-pooling
        #print('pooled_vector: ', pooled_vector.shape)
        pooled_vector = self.dropout(pooled_vector)
        output = self.classifier(pooled_vector)
        # сразу сделаем reshape 
        output = output.view(-1)
        #print('out: ', output.shape)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}


    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}
    
    def test_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        roc_auc = roc_auc_score(predictions, labels)
        self.log(f"roc_auc/Train", roc_auc, self.current_epoch)
        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Train", accuracy, self.current_epoch)
        
    def validation_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Val", accuracy, self.current_epoch)
        
    def test_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Test", accuracy, self.current_epoch)  


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )

def collar_classification(test_path, Path_2_model="/Users/egor/Desktop/Хак/model_vit_collars.pt"):
    def get_dataloader(test_path):
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        test_imgs = os.listdir(test_path)
        test_dataset = TestDatasetViT(test_path, test_imgs, feature_extractor)
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False)
        return test_loader

    # const
    n_classes = 1
    MODEL_NAME = 'google/vit-base-patch16-224-in21k'
    vit_model = ViTModel.from_pretrained(MODEL_NAME, return_dict=True, output_hidden_states=True)
    warmup_steps, total_training_steps = 26, 133
    
    # data
    test_loader = get_dataloader(test_path)
    
    # model
    model = CollarTagger(n_classes=n_classes, ViT_model=vit_model, n_warmup_steps=warmup_steps, 
                      n_training_steps=total_training_steps)
    
    # load weights
    model.load_state_dict(torch.load(Path_2_model))
    model.eval()

    # prediction
    prediction = []
    names = []
    for img_name, imgs, features in test_loader:
        names.extend(img_name)
        _, outputs = model(imgs, features, None)
        preds = ['false' if x > 0.0 else 'true' for x in outputs]
        prediction.extend(preds)

    return dict(zip(names, prediction))

collar_classification("/Users/egor/Desktop/Хак/Augmented_v4/test")