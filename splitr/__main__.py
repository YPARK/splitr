import os
import sys
import argparse
import mmutil

sys.path.insert(1, os.path.dirname(__file__))

from .util import *

###################
# local libraries #
###################

sys.path.insert(1, os.path.dirname(__file__))

def call_nb_vae(args) :

    import vae_nb

    if args.data is None:
        _log_msg("Need data to fit the model!")
        exit(1)

    if args.out is None:
        _log_msg("Provide the output header!")
        exit(1)

    model, latent_models, trace = vae_nb.run(args)
    vae_nb.write_results(model, latent_models, trace, args)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = "Train Negative Binomial VAE",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--task',
        default = 'vae',
        choices=['vae', 'deconv'],
        help = "a task to do",
    )

    parser.add_argument('--data', default = None, help = "matrix market data file")
    parser.add_argument('--columns_are_samples', default = True, help = "columns are samples")
    parser.add_argument('--covar', default = None, help = "matrix market data file" )
    parser.add_argument('--standardize_covar', default = True, help = "standardization of covariates" )
    parser.add_argument('--dlatent', default = "32,256,16", help = "latent dimension for encoding (mu)" )
    parser.add_argument('--dlibrary', default = "16,1", help = "latent dimension for encoding (library size)" )
    parser.add_argument('--batch', default = 100, type = int, help = "batch size" )
    parser.add_argument('--learning_rate', default = 1e-4, type = float, help = "learning rate" )
    parser.add_argument('--clip_value', default = 1e-2, type = float, help = "clip value" )
    parser.add_argument('--epochs', default = 1000, type = int, help = "number of epochs" )
    parser.add_argument('--iaf_trans', default = 4, type = int, help = "number of IAF transformations" )
    parser.add_argument('--dropout', default = 0.1, type = float, help = "Apply dropout to avoid over-fitting" )
    parser.add_argument('--spike', default = False, type = bool, help = "With spike-and-slab layer" )
    parser.add_argument('--kl_rate', default = 0.5, type = float, help = "KL term annealing rate" )
    parser.add_argument('--standardize_data', default = True, help = "standardization of data" )
    parser.add_argument('--std_target', default = 1e4, type = float, help = "The targeting count of standardization procedure" )
    parser.add_argument('--a0', default = 1e-4, type = float, help = "Minimum inverse overdispersion" )

    parser.add_argument(
        '--sample_cutoff',
        default = 0,
        type = float,
        help = "A minimum number of non-zero genes within each sample"
    )

    parser.add_argument(
        '--feature_cutoff',
        default = 0,
        type = float,
        help = "A minimum frequency of features across the retained samples"
    )

    parser.add_argument('--out', default=None)

    args = parser.parse_args()

    if args.task == 'vae':
        call_nb_vae(args)
    elif args.task == 'deconvolve':
        _log_msg('implement this')
    else:
        _log_msg('Undefined task: %s'%(args.task))
        exit(1)

    exit(0)
