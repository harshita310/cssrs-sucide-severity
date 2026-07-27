from utils.losses import OrdinalCrossEntropyLoss, build_criterion
from utils.trainer import TrainerConfig


def test_trainer_config_exposes_ordinal_distance_weight():
    cfg = TrainerConfig(ordinal_distance_weight=0.25)
    assert cfg.ordinal_distance_weight == 0.25


def test_build_criterion_returns_ordinal_ce_with_distance_weight():
    criterion = build_criterion(
        loss_type="ordinal_ce",
        ordinal_distance_weight=0.25,
        num_labels=7,
    )
    assert isinstance(criterion, OrdinalCrossEntropyLoss)
    assert criterion.distance_weight == 0.25
