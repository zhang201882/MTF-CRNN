from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

import numpy
import argparse
import textwrap

from dcase_framework.application_core import BinarySoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import setup_logging


class Task2AppCore(BinarySoundEventAppCore):
    pass


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(''' DCASE 2017 Task 2: Detection of rare sound events Baseline System ''')
    )

    # Setup argument handling
    parser.add_argument('-m', '--mode', choices=('dev', 'challenge'), default='challenge', help="Selector for system mode", required=False, dest='mode', type=str)
    parser.add_argument("-o", "--overwrite", choices=('true', 'false'), default='false', help="Overwrite exsiting files", dest="overwrite", required=False)
    parser.add_argument("-gpu", "--gpu", default=0, help="choose which gpu to use", required=True)

    # Parse arguments
    args = parser.parse_args()

    # set GPU number
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters/task2.defaults.yaml')

    # Initialize ParameterContainer
    params = ParameterContainer(
        project_base=os.path.dirname(os.path.realpath(__file__)),
        path_structure={
            'feature_extractor': ['dataset', 'feature_extractor.parameters.*'],
            'feature_normalizer': ['dataset', 'feature_extractor.parameters.*'],
            'learner': ['dataset', 'feature_extractor', 'feature_stacker', 'feature_normalizer', 'feature_aggregator', 'learner'],
            'recognizer': ['dataset', 'feature_extractor', 'feature_stacker', 'feature_normalizer', 'feature_aggregator', 'learner', 'recognizer'],
        }
    )

    # Load default parameters from a file
    params.load(filename=default_parameters_filename)

    # Process parameters
    params.process()
    
    # Force overwrite
    if args.overwrite == 'true':
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

    # Setup logging
    setup_logging(parameter_container=params['logging'])

    app = Task2AppCore(
        name='DCASE 2017::Detection of rare sound events / Baseline System',
        params=params,
        system_desc=params.get('description'),
        system_parameter_set_id=params.get('active_set'),
        setup_label='Development setup',
        log_system_progress=params.get_path('general.log_system_progress'),
        show_progress_in_console=params.get_path('general.print_system_progress'),
        use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
    )
    
    
    # Initialize application
    # ==================================================
    if params['flow']['initialize']:
        app.initialize()

    # Extract features for all audio files in the dataset
    # ==================================================
    if params['flow']['extract_features']:
        app.feature_extraction()

    # Prepare feature normalizers
    # ==================================================
    if params['flow']['feature_normalizer']:
        app.feature_normalization()
   
    # System training
    # ==================================================
    if params['flow']['train_system']:
        app.system_training()
    

    # System evaluation in development mode
    if not args.mode or args.mode == 'dev':

        # System testing
        # ==================================================
        if params['flow']['test_system']:
            app.system_testing()

        # System evaluation
        # ==================================================
        if params['flow']['evaluate_system']:
            app.system_evaluation()

    # System evaluation with challenge data
    elif args.mode == 'challenge':
        # Set dataset to testing set for challenge
        params['dataset']['method'] = 'challenge_test'

        # Process dataset again, move correct parameters from dataset_parameters
        params.process_method_parameters('dataset')

        if params['general']['challenge_submission_mode']:
            # If in submission mode, save results in separate folder for easier access
            params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

        challenge_app = Task2AppCore(
            name='DCASE 2017::Detection of rare sound events / Baseline System',
            params=params,
            system_desc=params.get('description'),
            system_parameter_set_id=params.get('active_set'),
            setup_label='Evaluation setup',
            log_system_progress=params.get_path('general.log_system_progress'),
            show_progress_in_console=params.get_path('general.print_system_progress'),
            use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
        )

        # Initialize application
        if params['flow']['initialize']:
            challenge_app.initialize()

        # Extract features for all audio files in the dataset
        if params['flow']['extract_features']:
            challenge_app.feature_extraction()

        # System testing
        if params['flow']['test_system']:
            if params['general']['challenge_submission_mode']:
                params['general']['overwrite'] = True

            challenge_app.system_testing(single_file_per_fold=True)

            if params['general']['challenge_submission_mode']:
                challenge_app.ui.line(" ")
                challenge_app.ui.line("Results for the challenge data are stored at ["+params['path']['recognizer_challenge_output']+"]")
                challenge_app.ui.line(" ")

        # System evaluation if not in challenge submission mode
        if params['flow']['evaluate_system']:
            challenge_app.system_evaluation(single_file_per_fold=True)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
