from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, Dice


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'jaccard': BinaryJaccardIndex(**kwargs),
            'dice': Dice(**kwargs),
        },
    )
