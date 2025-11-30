from datasets import load_dataset

ds = load_dataset("DScomp380/plant_village")
split_name = "train" if "train" in ds else list(ds.keys())[0]
labels = ds[split_name].features["label"].names

print(len(labels))
for i, name in enumerate(labels):
    print(i, name)
