
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
        self.correct += (output.argmax(dim=1) == label).sum().item()
        self.total += label.size()[0]
    
    def value(self):
        return self.correct / self.total
    
    @staticmethod
    def from_config(cfg):
        return Accuracy(**cfg)

class F1(BaseMetric):
    def __init__(self):
        super().__init__()
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, output, label):
        pred = output.argmax(dim=1)
        self.tp += ((pred == 1) & (label == 1)).sum().item()
        self.fp += ((pred == 1) & (label == 0)).sum().item()
        self.tn += ((pred == 0) & (label == 0)).sum().item()
        self.fn += ((pred == 0) & (label == 1)).sum().item()
    
    def value(self):
        if self.tp + self.fp == 0:
            precision = 0
        else:
            precision = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0:
            recall = 0
        else:
            recall = self.tp / (self.tp + self.fn)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def from_config(cfg):
        return F1(**cfg)

class Precision(BaseMetric):
    def __init__(self):
        super().__init__()
        self.tp = 0
        self.fp = 0

    def update(self, output, label):
        pred = output.argmax(dim=1)
        self.tp += ((pred == 1) & (label == 1)).sum().item()
        self.fp += ((pred == 1) & (label == 0)).sum().item()
    
    def value(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)
    
    @staticmethod
    def from_config(cfg):
        return Precision(**cfg)
    
class Recall(BaseMetric):
    def __init__(self):
        super().__init__()
        self.tp = 0
        self.fn = 0

    def update(self, output, label):
        pred = output.argmax(dim=1)
        self.tp += ((pred == 1) & (label == 1)).sum().item()
        self.fn += ((pred == 0) & (label == 1)).sum().item()
    
    def value(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)
    
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
    return "\n ".join([f"{k}: {v}" for k, v in metrics_dict.items()])


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