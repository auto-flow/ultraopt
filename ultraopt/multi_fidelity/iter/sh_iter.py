import numpy as np

from ultraopt.multi_fidelity.iter.base_iter import BaseIteration


class SuccessiveHalvingIteration(BaseIteration):

    def _advance_to_next_stage(self, config_ids, losses):
        """
            SuccessiveHalvingIteration simply continues the best based on the current loss.
        """
        ranks = np.argsort(np.argsort(losses))
        return ranks < self.num_configs[self.stage]
