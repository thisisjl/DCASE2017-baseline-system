import sys
import numpy
import argparse
import textwrap
import platform

from dcase_framework.utils import *
from dcase_framework.parameters import ParameterContainer

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

# from .choi2016 import MusicTaggerCRNN

def parse_args(argv):

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                DCASE 2017
                Task 1: Acoustic Scene Classification
                Example how to customize application
                ---------------------------------------------
                    Tampere University of Technology / Audio Research Group
                    Author:  Toni Heittola ( toni.heittola@tut.fi )

                System description
                    A system for acoustic scene classification, using DCASE 2013 Challenge evalution dataset.
                    Features: mean and std of centroid + zero crossing rate inside 1 second non-overlapping segments
                    Classifier: SVM

            '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    return parser.parse_args()

def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    args = parse_args(argv)

    # Load default parameters from a file
    default_parameters_filename = 'parameters.yaml'

    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True
            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

        # app = CustomAppCore(name='DCASE 2017::Acoustic Scene Classification / Baseline System',
        #                     params=params,
        #                     system_desc=params.get('description'),
        #                     system_parameter_set_id=params.get('active_set'),
        #                     setup_label='Development setup',
        #                     log_system_progress=params.get_path('general.log_system_progress'),
        #                     show_progress_in_console=params.get_path('general.print_system_progress'),
        #                     use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
        #                     )

        # Show parameter set list and exit
        # if args.show_set_list:
        #     params_ = ParameterContainer(
        #         project_base=os.path.dirname(os.path.realpath(__file__))
        #     ).load(filename=default_parameters_filename)
        #
        #     if args.parameter_override:
        #         # Override parameters from a file
        #         params_.override(override=args.parameter_override)
        #     if 'sets' in params_:
        #         app.show_parameter_set_list(set_list=params_['sets'])
        #
        #     return

        # # Show dataset list and exit
        # if args.show_dataset_list:
        #     app.show_dataset_list()
        #     return

        # Show system parameters
        # if params.get_path('general.log_system_parameters') or args.show_parameters:
        #     app.show_parameters()

        # Show evaluated systems
        # if args.show_eval:
        #     app.show_eval()
        #     return

        # Initialize application
        # ==================================================
        # if params['flow']['initialize']:
        #     app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        # if params['flow']['extract_features']:
        #     app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        # if params['flow']['feature_normalizer']:
        #     app.feature_normalization()

        # System training
        # ==================================================
        # if params['flow']['train_system']:
        #     app.system_training()

        # System evaluation
        # if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            # if params['flow']['test_system']:
            #     app.system_testing()

            # System evaluation
            # ==================================================
            # if params['flow']['evaluate_system']:
            #     app.system_evaluation()

        # System evaluation in challenge mode


    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
