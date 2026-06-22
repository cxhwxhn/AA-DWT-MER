import torch
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import WeightedRandomSampler


def train(args, model, train_dataset, test_dataset=None, train_log_file=None, test_log_file=None):
    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty. LOSO training needs at least two subjects.")
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("test_dataset is empty.")
    if args.eval_interval < 1:
        raise ValueError("eval_interval must be greater than 0.")
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be greater than 0.")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze_backbone_epochs must not be negative.")
    if not 0 < args.backbone_lr_scale <= 1:
        raise ValueError("backbone_lr_scale must be in (0, 1].")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )
    # criterion = torch.nn.CrossEntropyLoss(    )
    optimizer = torch.optim.SGD(
        [
            {
                "params": model.backbone.parameters(),
                "lr": args.lr * args.backbone_lr_scale,
            },
            {"params": model.enhance.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        momentum=args.momentum,
        weight_decay=args.wdecay,
    )

    # optimizer = torch.optim.SGD(
    #     [
    #         {
    #             "params": model.backbone.parameters(),
    #             "lr": args.lr * args.backbone_lr_scale,
    #         },
            
    #     ],
    #     momentum=args.momentum,
    #     weight_decay=args.wdecay,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_lr,
    )
    train_sampler, class_counts = build_weighted_sampler(train_dataset, args.cls)
    print("train_class_counts:{}".format(class_counts.tolist()))
    train_dataloader = DataLoader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    final_acc = -1
    final_uar = -1
    final_cm = init_confusion_matrix(args.cls)

    if args.freeze_backbone_epochs > 0:
        set_requires_grad(model.backbone, False)

    for epoch in range(1, args.num_epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_requires_grad(model.backbone, True)
            print("Backbone unfrozen at epoch {}.".format(epoch))

        model.train()
        if epoch <= args.freeze_backbone_epochs:
            model.backbone.eval()
        total_samples = 0
        correct_samples = 0
        epoch_loss = 0.0

        for image, label in train_dataloader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred_mer = model(image)
            loss = criterion(pred_mer, label)
            _, pred = torch.max(pred_mer, dim=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            correct_samples += pred.eq(label).sum().item()
            total_samples += len(label)
            epoch_loss += loss.item() * len(label)

        scheduler.step()

        should_report = (epoch % args.eval_interval == 0) or (epoch == args.num_epochs)

        if should_report:
            avg_loss = epoch_loss / max(total_samples, 1)
            acc = correct_samples * 100.0 / max(total_samples, 1)
            print("-----epoch:{}/{}-----".format(epoch, args.num_epochs))
            backbone_lr = optimizer.param_groups[0]["lr"]
            head_lr = optimizer.param_groups[2]["lr"]
            print(
                "train_loss:{}\ttrain_acc:{}%".format(
                    avg_loss, acc,
                )
            )
            print("=========================")
            if train_log_file is not None:
                train_log_file.writelines(
                    "-----epoch:{}/{}-----\n".format(epoch, args.num_epochs)
                )
                train_log_file.writelines(
                    "train_loss:{}\ttrain_acc:{}\tbackbone_lr:{}\thead_lr:{}\n".format(
                        avg_loss, acc, backbone_lr, head_lr
                    )
                )
                train_log_file.writelines("=========================\n")
                train_log_file.flush()

            test_acc, cm = evaluate(
                args=args,
                model=model,
                epoch=epoch,
                test_dataset=test_dataset,
                test_log_file=test_log_file,
            )
            _, _, test_uar, _, _ = calculate_metrics(cm)
          
            if test_log_file is not None:
                test_log_file.writelines("uar:{}\n".format(test_uar))
                test_log_file.flush()

            if test_uar > final_uar:
                final_uar = test_uar
                final_acc = test_acc
                final_cm = cm
                torch.save(model.state_dict(), args.save_path)
                
                if test_log_file is not None:
                    test_log_file.writelines(
                        "best model saved by UAR: {} (uar:{}, acc:{}%)\n".format(
                            args.save_path, final_uar, final_acc
                        )
                    )
                    test_log_file.flush()

    return final_acc, final_cm


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def evaluate(args, model, epoch, test_dataset, test_log_file):
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("test_dataset is empty.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_samples = 0
    correct_samples = 0
    total_loss = 0.0
    confusion_matrix = init_confusion_matrix(args.cls)
    criterion = torch.nn.CrossEntropyLoss()
    test_dataloader = DataLoader.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            image, label = item
            image = image.to(device)
            label = label.to(device)

            pred_mer = model(image)
            loss = criterion(pred_mer, label)
            _, pred = torch.max(pred_mer, dim=1)

            pred_list = pred.cpu().numpy().tolist()
            label_list = label.cpu().numpy().tolist()

            correct_sample, confusion_matrix = cal_corr(
                label_list,
                pred_list,
                confusion_matrix,
            )
            correct_samples += correct_sample
            total_samples += len(label_list)
            total_loss += loss.item() * len(label_list)

        acc = correct_samples * 100.0 / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)
        print("-----epoch:{}-----".format(epoch))
        print("acc:{}%".format(acc))
        print("loss:{}".format(avg_loss))
        target_distribution, pred_distribution = confusion_distribution(confusion_matrix)
   

        if test_log_file is not None:
            test_log_file.writelines("\n-----epoch:{}-----\n".format(epoch))
            test_log_file.writelines("acc:{}\n".format(acc))
            test_log_file.writelines("loss:{}\n".format(avg_loss))
            test_log_file.writelines("target_distribution:{}\n".format(target_distribution))
            test_log_file.writelines("pred_distribution:{}\n".format(pred_distribution))
            test_log_file.writelines("confusion_matrix:\n{}\n".format(confusion_matrix))
            test_log_file.flush()

    return acc, confusion_matrix


def init_confusion_matrix(cls):
    return [[0 for _ in range(cls)] for _ in range(cls)]


def confusion_distribution(confusion_matrix):
    cm = torch.tensor(confusion_matrix, dtype=torch.long)
    target_distribution = cm.sum(dim=1).tolist()
    pred_distribution = cm.sum(dim=0).tolist()
    return target_distribution, pred_distribution


def build_weighted_sampler(dataset, cls):
    counts = torch.zeros(cls, dtype=torch.float64)
    labels = []
    for row in dataset.rows:
        label = dataset.label_to_idx[row["label"]]
        labels.append(label)
        counts[label] += 1

    class_sample_weights = counts.clamp_min(1.0).reciprocal()
    sample_weights = torch.tensor(
        [class_sample_weights[label].item() for label in labels],
        dtype=torch.float64,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, counts.to(dtype=torch.long)


def cal_corr(label_list, pred_list, confusion_matrix):
    corr = 0
    for label, pred in zip(label_list, pred_list):
        confusion_matrix[label][pred] += 1
        if label == pred:
            corr += 1
    return corr, confusion_matrix


def calculate_metrics(confusion_matrix):
    cm = torch.tensor(confusion_matrix, dtype=torch.float32)
    correct = torch.diag(cm).sum()
    total = cm.sum().clamp_min(1)
    acc = (correct / total).item()

    recall = torch.diag(cm) / cm.sum(dim=1).clamp_min(1)
    precision = torch.diag(cm) / cm.sum(dim=0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    class_support = cm.sum(dim=1)
    weight = class_support / class_support.sum().clamp_min(1)
    war = acc
    uar = recall.mean().item()
    wf1 = (f1 * weight).sum().item()
    uf1 = f1.mean().item()
    return acc, war, uar, wf1, uf1
