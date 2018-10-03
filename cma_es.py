import cma
import math
import numpy as np

from datetime import datetime
from print_log import PrintLog


class CmaEs:

    def __init__(self, func, para):
        # "para" is the diction of parameters which needs to be optimized.
        # "para" --> {'x_1': (0,10), 'x_2': (-1, 1),..., 'x_n':(Lower_bound, Upper_bound)}

        self.f = func
        self.begin_time = datetime.now()
        self.timestamps_list = []
        self.target_list = []
        self.parameters_list = []
        self.pop_list = []
        self.keys = list(para.keys())
        self.bounds = np.array(list(para.values()), dtype=np.float)
        self.dim = len(self.keys)
        self.plog = PrintLog(self.keys)
        self.para_value = np.empty((1, self.dim))
        self.plog.print_header(initialization=True)
        for col, (lower, upper) in enumerate(self.bounds):
            self.para_value.T[col] = np.random.RandomState().uniform(lower, upper)
        self.para_value = self.para_value.ravel().tolist()

    def evaluate(self, input):
        result = self.f(input[0], input[1])
        return -result

    def run(self, max_iter=20, pop_size=10, sigma=0.5):

        # "sigma0" is the initial standard deviation.
        # The problem variables should have been scaled, such that a single standard deviation
        # on all variables is useful and the optimum is expected to lie within about `x0` +- ``3*sigma0``.
        # See also options 'scaling_of_variables'. Often one wants to check for solutions close to the initial point.
        # This allows, for example, for an easier check of consistency of the
        # objective function and its interfacing with the optimizer.
        # In this case, a much "smaller" 'sigma0' is advisable.
        sigma_0 = sigma

        # "conf_para" is used to configure the parameters in CMA-ES algorithm
        # "conf_para" --> {'maxiter': 20, 'popsize': 20}
        conf_para = {'maxiter': max_iter, 'popsize': pop_size}

        es = cma.CMAEvolutionStrategy(self.para_value, sigma_0, conf_para)

        self.plog.print_header(initialization=False)

        while not es.stop():
            solutions = es.ask()
            self.pop_list.append(solutions)
            es.tell(solutions, [self.evaluate(x) for x in solutions])
            #        es.tell(*es.ask_and_eval(f))
            #        es.disp()
            res = es.result
            #        metric = f(**params_dic)
            self.parameters_list.append(res[0].tolist())
            self.target_list.append(- res[1])
            elapse_time = (datetime.now() - self.begin_time).total_seconds()
            self.timestamps_list.append(elapse_time)
            #        print("The best candidate: ", res[0])
            #        print("The best result: ", res[1])
            self.plog.print_step(res[0], - res[1])

        return self.timestamps_list, self.target_list, self.parameters_list, self.pop_list