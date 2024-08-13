import math
from torch.optim.lr_scheduler import LRScheduler

class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, initial_lr: float = 1e-10, last_iter: int = -1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_iter)

    def get_lr(self):
        if self._step_count <= self.warmup_iters:
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self._step_count / self.warmup_iters
                for base_lr in self.base_lrs]
        else:
            cos_iter = self._step_count - self.warmup_iters
            cos_max_iter = self.max_iters - self.warmup_iters
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [base_lr * (1 + math.cos(cos_theta)) / 2 for base_lr in self.base_lrs]
            return cos_lr
