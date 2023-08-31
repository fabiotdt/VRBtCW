from datetime import datetime
from typing import List

from pytorch_lightning.loggers import CSVLogger
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch import nn, softmax
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
batch_size = 64
# Initialize an empty dictionary
imagenet_labels_to_classes = {}

# fix all seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Read the file and populate the dictionary
with open("imagenet_class_labels.txt", "r") as f:
    for line in f.readlines():
        elements = line.split("\t")
        key = elements[0]
        value = elements[1]
        imagenet_labels_to_classes[key] = value.strip()

BASIC_STRING = "A photo of a "


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        # Initialize the parent class
        super(CustomImageFolder, self).__init__(root, transform=transform)

        # Initialize an empty list to store the textual class labels
        self.class_text_labels = []

        # Populate the class_text_labels list
        for class_name, class_idx in self.class_to_idx.items():
            if "imagenet" in root.lower():
                # Convert using the imagenet_classes dictionary
                self.class_text_labels.append(
                    imagenet_labels_to_classes[class_name]
                )
            else:
                # Use the original class name
                class_name = class_name.split(".")[-1].lower()
                self.class_text_labels.append(class_name)
        self.text_features = None
        self.texts = None

    def get_text_features(self, emebd_function) -> torch.Tensor:
        if self.text_features is not None:
            return self.text_features
        self.text_features = []
        for class_text_label in self.class_text_labels:
            text_feature = emebd_function(BASIC_STRING + class_text_label)
            self.text_features.append(text_feature)

        # stack the tensor from a list of (1, 512) to (N, 512)
        self.text_features = torch.stack(self.text_features).squeeze()
        return self.text_features

    def get_texts(self) -> List[str]:
        if self.texts is not None:
            return self.texts
        self.texts = []
        for class_text_label in self.class_text_labels:
            text_feature = "A photo of a " + class_text_label
            self.texts.append(text_feature)
        return self.texts

    def get_texts_from_tensor(self, tensor_indexes) -> List[str]:
        return_list = []
        texts = self.get_texts()
        for index in tensor_indexes:
            return_list.append(texts[index])
        return return_list


imagenet_dir = "Imagenet_o20"
calltech_dir = "Calltech_24"

# Define transform for datasets
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
transform = lambda x: processor(images=x, return_tensors="pt")[
    "pixel_values"
].squeeze()
# transform = transforms.Compose(
#     [transforms.Resize((224, 224)), transforms.ToTensor()]
# )

# Test Dataset (ImageNet subset)
test_dataset = CustomImageFolder(imagenet_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Train Dataset (Caltech256)
train_dataset = CustomImageFolder(calltech_dir, transform=transform)
# Train-Val-Test split for imagenet_24
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
calltech_train, calltech_val = random_split(
    train_dataset, [train_size, val_size]
)
train_loader = DataLoader(calltech_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(calltech_val, batch_size=batch_size)


class FineTuneCLIP(LightningModule):
    def __init__(self):
        super(FineTuneCLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def get_train_dataset(self):
        return self.trainer.train_dataloader.dataset.dataset

    def get_val_dataset(self):
        return self.trainer.val_dataloaders[0].dataset.dataset

    def get_single_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        # Forward pass through the model
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings

    def get_single_image_embedding(self, images):
        # inputs = self.processor(images=images, return_tensors="pt")

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model.get_image_features(images)
        return outputs

    def forward(self, images, texts):
        return self.model(images, texts)

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        texts = self.get_train_dataset().get_texts_from_tensor(labels)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs["pixel_values"] = images

        outputs = self.model(**inputs, return_loss=True)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        texts = (
            self.trainer.val_dataloaders.dataset.dataset.get_texts_from_tensor(
                labels
            )
        )
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs["pixel_values"] = images

        with torch.no_grad():
            outputs = self.model(**inputs, return_loss=True)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        probs = outputs.logits_per_image.softmax(dim=1)
        predictions = torch.argmax(probs, dim=1)
        val_acc = accuracy_score(labels.cpu(), predictions.cpu())
        self.log(
            "val_acc",
            val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "val_acc": val_acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        image_features = self.get_single_image_embedding(images)
        text_features_for_all_classes = self.trainer.test_dataloaders[
            0
        ].dataset.get_text_features(self.get_single_text_embedding)

        # normalized features
        image_features = image_features / image_features.norm(
            p=2, dim=-1, keepdim=True
        )
        text_features_for_all_classes = (
            text_features_for_all_classes
            / text_features_for_all_classes.norm(p=2, dim=-1, keepdim=True)
        )
        similarities = torch.matmul(
            text_features_for_all_classes, image_features.t()
        ).t()
        predictions = torch.argmax(similarities, dim=1)

        test_acc = accuracy_score(labels.cpu(), predictions.cpu())
        self.log(
            "test_acc",
            test_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"test_acc": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-3,
        )
        return optimizer


class FineTuneCLIP_Entropy(FineTuneCLIP):
    def get_test_texts(self):
        return test_dataset.get_texts()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Original task loss
        texts = self.get_train_dataset().get_texts_from_tensor(labels)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs["pixel_values"] = images
        outputs = self.model(**inputs, return_loss=True)
        original_loss = outputs.loss

        # Step 2: Get Text Features for Test Classes
        test_texts = self.get_test_texts()
        inputs = self.processor(
            text=test_texts, return_tensors="pt", padding=True
        )
        inputs["pixel_values"] = images
        outputs_test = self.model(**inputs)

        # Step 4: Compute Entropy Loss
        softmax_output = softmax(outputs_test.logits_per_image, dim=1)
        entropy_loss = -torch.mean(
            torch.sum(softmax_output * torch.log(softmax_output + 1e-6), dim=1)
        )

        # Step 5: Combine the Two Losses
        # Assuming lambda_factor is a hyperparameter to weight the importance of the entropy loss
        lambda_factor = 0.1
        final_loss = original_loss + lambda_factor * entropy_loss

        self.log(
            "train_loss",
            final_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": final_loss}


# Initialize the model
# model = FineTuneCLIP()
model = FineTuneCLIP_Entropy()

# Early stopping based on validation loss
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.0, patience=3, verbose=True, mode="min"
)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_logger = CSVLogger("logs", name=f"fine-tune-clip_{current_time}")
# Initialize the Model Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Replace with your validation metric
    mode="min",  # Mode can be 'max' or 'min' depending on the metric
    save_top_k=1,  # Save only the best model
    filename="{epoch}-{val_loss:.2f}",  # Filename format
    verbose=True,  # Log when a better model is found
)

# Trainer setup
trainer = Trainer(
    max_epochs=30,
    callbacks=[early_stop_callback, checkpoint_callback],
    accelerator=device,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    logger=csv_logger,
    log_every_n_steps=10,
)

# trainer.test(model, dataloaders=[test_dataloader])

trainer.fit(model, train_loader, val_loader)

trainer.test(model, dataloaders=[test_dataloader])
