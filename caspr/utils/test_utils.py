import numpy as np

#
# testing helper functions
#

class TestStatTracker():
    '''
    Tracks statistics of interest while running inference on the
    validation or test sets
    '''
    def __init__(self):
        self.loss_sum = 0.0
        self.total_loss_count = 0

        self.cnf_err_sum = 0.0
        self.cnf_err_count = 0

        self.tnocs_pos_err_sum = 0.0
        self.tnocs_pos_err_count = 0

        self.tnocs_time_err_sum = 0.0
        self.tnocs_time_err_count = 0

        self.nfe_sum = np.array([0.0, 0.0])

    def record_stats(self, loss_scalar, cnf_err, tnocs_pos_err, tnocs_time_err, nfe):
        '''
        Takes in a loss value and numpy arrays of various losses/errors and adds to
        the current sum. Updates counts automatically.
        '''
        self.loss_sum += loss_scalar
        self.total_loss_count += 1

        self.cnf_err_sum += np.sum(cnf_err)
        self.cnf_err_count += cnf_err.shape[0]*cnf_err.shape[1]*cnf_err.shape[2]

        self.tnocs_pos_err_sum += np.sum(tnocs_pos_err)
        self.tnocs_pos_err_count += tnocs_pos_err.shape[0]

        self.tnocs_time_err_sum += np.sum(tnocs_time_err)
        self.tnocs_time_err_count += tnocs_time_err.shape[0]

        self.nfe_sum += nfe

    def get_mean_stats(self):
        ''' Return mean for all values '''
        total_loss_out = self.loss_sum / self.total_loss_count

        cnf_err_out = self.cnf_err_sum / self.cnf_err_count
        tnocs_pos_err_out = self.tnocs_pos_err_sum / self.tnocs_pos_err_count
        tnocs_time_err_out = self.tnocs_time_err_sum / self.tnocs_time_err_count
        mean_nfe = self.nfe_sum / self.total_loss_count

        return total_loss_out, cnf_err_out, tnocs_pos_err_out, tnocs_time_err_out, mean_nfe