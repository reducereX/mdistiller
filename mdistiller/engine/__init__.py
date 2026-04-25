from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, SFWSupConTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "SFWSupCon": SFWSupConTrainer,
}
