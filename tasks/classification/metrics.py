SUPPORTED_METRICS = ["accuracy", "f1", "precision", "recall"]

class BaseMetric():
    def __init__(self):
        pass

    def update(self, output, label):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0

    def update(self, output, label):
        pred = output.argmax(dim=1)
        self.correct += (pred == label).sum().item()
        self.total += label.size(0)

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    @staticmethod
    def from_config(cfg):
        return Accuracy(**cfg)

class F1(BaseMetric):
    def __init__(self, num_classes=2, average='macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.tp = [0] * num_classes
        self.fp = [0] * num_classes
        self.fn = [0] * num_classes
        self.support = [0] * num_classes  # Number of true instances for each class

    def update(self, output, label):
        pred = output.argmax(dim=1)
        for i in range(self.num_classes):
            self.tp[i] += ((pred == i) & (label == i)).sum().item()
            self.fp[i] += ((pred == i) & (label != i)).sum().item()
            self.fn[i] += ((pred != i) & (label == i)).sum().item()
            self.support[i] += (label == i).sum().item()

    def value(self):
        precision = []
        recall = []
        f1 = []
        for i in range(self.num_classes):
            tp = self.tp[i]
            fp = self.fp[i]
            fn = self.fn[i]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * p * r / (p + r) if (p + r) > 0 else 0
            precision.append(p)
            recall.append(r)
            f1.append(f1_score)

        if self.average == 'macro':
            return sum(f1) / self.num_classes
        elif self.average == 'micro':
            total_tp = sum(self.tp)
            total_fp = sum(self.fp)
            total_fn = sum(self.fn)
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0
        elif self.average == 'weighted':
            total_support = sum(self.support)
            weighted_f1 = sum(f * s for f, s in zip(f1, self.support)) / total_support if total_support > 0 else 0
            return weighted_f1
        else:
            raise ValueError(f"Unsupported average type: {self.average}")

    @staticmethod
    def from_config(cfg):
        return F1(**cfg)

class Precision(BaseMetric):
    def __init__(self, num_classes=2, average='macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.tp = [0] * num_classes
        self.fp = [0] * num_classes
        self.support = [0] * num_classes

    def update(self, output, label):
        pred = output.argmax(dim=1)
        for i in range(self.num_classes):
            self.tp[i] += ((pred == i) & (label == i)).sum().item()
            self.fp[i] += ((pred == i) & (label != i)).sum().item()
            self.support[i] += (label == i).sum().item()

    def value(self):
        precision = []
        for i in range(self.num_classes):
            tp = self.tp[i]
            fp = self.fp[i]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision.append(p)

        if self.average == 'macro':
            return sum(precision) / self.num_classes
        elif self.average == 'micro':
            total_tp = sum(self.tp)
            total_fp = sum(self.fp)
            return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        elif self.average == 'weighted':
            total_support = sum(self.support)
            weighted_precision = sum(p * s for p, s in zip(precision, self.support)) / total_support if total_support > 0 else 0
            return weighted_precision
        else:
            raise ValueError(f"Unsupported average type: {self.average}")

    @staticmethod
    def from_config(cfg):
        return Precision(**cfg)

class Recall(BaseMetric):
    def __init__(self, num_classes=2, average='macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.tp = [0] * num_classes
        self.fn = [0] * num_classes
        self.support = [0] * num_classes

    def update(self, output, label):
        pred = output.argmax(dim=1)
        for i in range(self.num_classes):
            self.tp[i] += ((pred == i) & (label == i)).sum().item()
            self.fn[i] += ((pred != i) & (label == i)).sum().item()
            self.support[i] += (label == i).sum().item()

    def value(self):
        recall = []
        for i in range(self.num_classes):
            tp = self.tp[i]
            fn = self.fn[i]
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall.append(r)

        if self.average == 'macro':
            return sum(recall) / self.num_classes
        elif self.average == 'micro':
            total_tp = sum(self.tp)
            total_fn = sum(self.fn)
            return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        elif self.average == 'weighted':
            total_support = sum(self.support)
            weighted_recall = sum(r * s for r, s in zip(recall, self.support)) / total_support if total_support > 0 else 0
            return weighted_recall
        else:
            raise ValueError(f"Unsupported average type: {self.average}")

    @staticmethod
    def from_config(cfg):
        return Recall(**cfg)

MODULE_NAME = {
    "accuracy": Accuracy,
    "f1": F1,
    "precision": Precision,
    "recall": Recall
}

def beautify(metrics_dict):
    return "\n".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])

def build(metric_name, cfg):
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    module = MODULE_NAME[metric_name]
    return module.from_config(cfg)

__all__ = [
    "build",
    "BaseMetric",
    "Accuracy",
    "F1",
    "Precision",
    "Recall"
]