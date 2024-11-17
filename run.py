from utils.get_args import parse_args
from experiments.experiment import Experiment


def main():
    args = parse_args()

    experiment = Experiment(args)

    for ii in range(args.itr):
        setting = '{}_{}_pca{}{}_crop{}_scale{}_rotate{}_filp{}_Exp_{}'.format(
            args.model,
            args.data,
            args.use_pca,
            args.pca_components,
            args.use_crop,
            args.use_scale,
            args.use_rotation,
            args.use_flip,
            ii
        )
        experiment.run(setting)

if __name__ == "__main__":
    main()
