from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice


def get_metrics() -> MetricCollection:
    return MetricCollection(
        {
            'jaccard': BinaryJaccardIndex(),
            'dice': Dice()
            # 'precision': Precision(**kwargs),
            # 'recall': Recall(**kwargs),
            # 'accuracy': Accuracy(**kwargs),
        },
    )
