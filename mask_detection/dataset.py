# -*- coding: utf-8 -*-
# Principal packages
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Commented out IPython magic to ensure Python compatibility.
# Drive Loading
# from google.colab import drive
# drive.mount('/gdrive')
# %cd /gdrive
drive_path = ""
BATCH_SIZE = 64
dataset_path = "mask_dataset"
dataset_path = drive_path + dataset_path

output_path = "mask_detector/"
output_path = drive_path + output_path

# the training transforms
train_transform = transforms.Compose(
    [
        transforms.CenterCrop(200),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # )
    ]
)

# the validation transforms
valid_transform = transforms.Compose(
    [
        transforms.CenterCrop(20),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

print("Loading images")

# Create datasets
train_dataset = datasets.ImageFolder(
    root=dataset_path + "/train", transform=train_transform
)
valid_dataset = datasets.ImageFolder(
    root=dataset_path + "/validation", transform=valid_transform
)

# Create data loader
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    pin_memory=True)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    pin_memory=True)

# Display image and label.
# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze().permute(1,2,0)
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

# print("Visualization of the test data")

# labels_map = {
#     0: "Mask",
#     1: "No mask",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
#     img, label = train_dataset[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.permute(1,2,0))
# plt.show()

#  def show_train_repartition():
#     data = {}
#     for i in train_labels:
#         curr_count = 0
#         for j in train_labels:
#             if i == j:
#                 curr_count+=1

#         for index, name in ethnicities.items():
#             if(index == i):
#                 data[name] = curr_count
#                 break

#     fig = plt.figure(figsize = (10, 5))

#     plt.bar(list(data.keys()), list(data.values()), color="#777777")

#     plt.show()

# show_train_repartition()
