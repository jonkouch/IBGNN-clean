from ibgnn.parser import ibgnn_parse_arguments
from ibgnn.experiments import NNExperiment


if __name__ == '__main__':
    args = ibgnn_parse_arguments()
    NNExperiment(args=args).run() 