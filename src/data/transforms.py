from torchvision import transforms

def get_transforms(img_size=256, normalize=True, augment=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = [transforms.Resize((img_size, img_size))]
    if augment:
        train_tf += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=15),
        ]
    train_tf += [transforms.ToTensor()]
    if normalize:
        train_tf += [transforms.Normalize(mean, std)]
    train = transforms.Compose(train_tf)

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
    ])

    return train, eval_tf
