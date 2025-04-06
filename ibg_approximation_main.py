from ibg_approximation.parser import ibg_approx_parse_arguments
from ibg_approximation.experiments import IBGApproxExperiment

if __name__ == '__main__':
    args = ibg_approx_parse_arguments()
    IBGApproxExperiment(args=args).run()
