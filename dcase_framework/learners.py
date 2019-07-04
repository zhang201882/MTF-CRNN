from __future__ import print_function, absolute_import
from six import iteritems

import sys
import numpy as np
import logging
import random
import warnings
import copy

from .pytorch_utils import PytorchMixin, CRNN_Reverse_Pytorch, SegmentedDataset, kaiming_init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from .files import DataFile
from .containers import ContainerMixin, DottedDict
from .features import FeatureContainer
from .metadata import EventRoll
#from .data import DataSequencer
#from .utils import get_class_inheritors
#from .recognizers import SceneRecognizer, EventRecognizer


def event_detector_factory(*args, **kwargs):
    if kwargs.get('method', None) == 'pytorch_crnn':
        return EventDetectorCRNN_Pytorch(*args, **kwargs)    
    else:
        raise ValueError('{name}: Invalid EventDetector method [{method}]'.format(name='event_detector_factory', method=kwargs.get('method', None)))


class LearnerContainer(DataFile, ContainerMixin):
    valid_formats = ['cpickle']

    def __init__(self, *args, **kwargs):
        """
            Constructor

            Parameters
            ----------
            method : str
                Method label
                Default value "None"
            class_labels : list of strings
                List of class labels
                Default value "[]"
            params : dict or DottedDict
                Parameters
            feature_masker : FeatureMasker or class inherited from FeatureMasker
                Feature masker instance
                Default value "None"
            feature_normalizer : FeatureNormalizer or class inherited from FeatureNormalizer
                Feature normalizer instance
                Default value "None"
            feature_stacker : FeatureStacker or class inherited from FeatureStacker
                Feature stacker instance
                Default value "None"
            feature_aggregator : FeatureAggregator or class inherited from FeatureAggregator
                Feature aggregator instance
                Default value "None"
            logger : logging
                Instance of logging
                Default value "None"
            disable_progress_bar : bool
                Disable progress bar in console
                Default value "False"
            log_progress : bool
                Show progress in log.
                Default value "False"
            show_extra_debug : bool
                Show extra debug information
                Default value "True"
        """
        super(LearnerContainer, self).__init__({
            'method': kwargs.get('method', None),
            'class_labels': kwargs.get('class_labels', []),
            'params': DottedDict(kwargs.get('params', {})),
            'feature_masker': kwargs.get('feature_masker', None),
            'feature_normalizer': kwargs.get('feature_normalizer', None),
            'feature_stacker': kwargs.get('feature_stacker', None),
            'feature_aggregator': kwargs.get('feature_aggregator', None),
            'model': kwargs.get('model', {}),
            'learning_history': kwargs.get('learning_history', {}),
        }, *args, **kwargs)

        # Set randomization seed
        if self.params.get_path('seed') is not None:
            self.seed = self.params.get_path('seed')
        elif self.params.get_path('parameters.seed') is not None:
            self.seed = self.params.get_path('parameters.seed')
        elif kwargs.get('seed', None):
            self.seed = kwargs.get('seed')
        else:
            epoch = datetime.utcfromtimestamp(0)
            unix_now = (datetime.now() - epoch).total_seconds() * 1000.0
            bigint, mod = divmod(int(unix_now) * 1000, 2**32)
            self.seed = mod

        self.logger = kwargs.get('logger',  logging.getLogger(__name__))
        self.disable_progress_bar = kwargs.get('disable_progress_bar',  False)
        self.log_progress = kwargs.get('log_progress',  False)
        self.show_extra_debug = kwargs.get('show_extra_debug', True)

    @property
    def class_labels(self):
        """
            Class labels

            Returns
            -------
            list of strings
                List of class labels in the model
        """
        return sorted(self.get('class_labels', None))

    @class_labels.setter
    def class_labels(self, value):
        self['class_labels'] = value

    @property
    def method(self):
        """
            Learner method label

            Returns
            -------
            str
                Learner method label
        """
        return self.get('method', None)

    @method.setter
    def method(self, value):
        self['method'] = value

    @property
    def params(self):
        """
            Parameters

            Returns
            -------
            DottedDict
                Parameters
        """
        return self.get('params', None)

    @params.setter
    def params(self, value):
        self['params'] = value

    @property
    def feature_masker(self):
        """
            Feature masker instance

            Returns
            -------
            FeatureMasker
        """
        return self.get('feature_masker', None)

    @feature_masker.setter
    def feature_masker(self, value):
        self['feature_masker'] = value

    @property
    def feature_normalizer(self):
        """
            Feature normalizer instance

            Returns
            -------
            FeatureNormalizer
        """
        return self.get('feature_normalizer', None)

    @feature_normalizer.setter
    def feature_normalizer(self, value):
        self['feature_normalizer'] = value

    @property
    def feature_stacker(self):
        """
            Feature stacker instance

            Returns
            -------
            FeatureStacker
        """
        return self.get('feature_stacker', None)

    @feature_stacker.setter
    def feature_stacker(self, value):
        self['feature_stacker'] = value

    @property
    def feature_aggregator(self):
        """
            Feature aggregator instance

            Returns
            -------
            FeatureAggregator
        """
        return self.get('feature_aggregator', None)

    @feature_aggregator.setter
    def feature_aggregator(self, value):
        self['feature_aggregator'] = value

    @property
    def model(self):
        """
            Acoustic model

            Returns
            -------
            model
        """
        return self.get('model', None)

    @model.setter
    def model(self, value):
        self['model'] = value

    def set_seed(self, seed=None):
        """
            Set randomization seeds

            Returns
            -------
            nothing
        """
        if seed is None:
            seed = self.seed

        np.random.seed(seed)
        random.seed(seed)

    @property
    def learner_params(self):
        """
            Get learner parameters from parameter container

            Returns
            -------
            DottedDict
                Learner parameters
        """
        if 'parameters' in self['params']:
            parameters = self['params']['parameters']
        else:
            parameters = self['params']
        return DottedDict({k: v for k, v in parameters.items() if not k.startswith('_')})

    def _get_input_size(self, data):
        input_shape = None
        for audio_filename in data:
            if not input_shape:
                input_shape = data[audio_filename].feat[0].shape[1]
            elif input_shape != data[audio_filename].feat[0].shape[1]:
                message = '{name}: Input size not coherent.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise ValueError(message)
        return input_shape


class EventDetector(LearnerContainer):
    """
        Event detector (Frame classifier / Multi-class - Multi-label)
    """

    def _get_target_matrix_dict(self, data, annotations):

        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            # Create event roll
            event_roll = EventRoll(metadata_container=annotations[audio_filename],
                                   label_list=self.class_labels,
                                   time_resolution=self.params.get_path('hop_length_seconds')
                                   )
            # Pad event roll to full length of the signal
            activity_matrix_dict[audio_filename] = event_roll.pad(length=data[audio_filename].shape[0])

        return activity_matrix_dict

    def _generate_validation(self, annotations, validation_type='generated_scene_location_event_balanced', valid_percentage=0.20, seed=None):

        self.set_seed(seed=seed)
        validation_files = []

        if self.show_extra_debug:
            print('\n\n')
            self.logger.debug('  Validation')

        if validation_type == 'generated_scene_location_event_balanced':
            # Get training data per scene label
            annotation_data = {}
            for audio_filename in sorted(list(annotations.keys())):
                scene_label = annotations[audio_filename][0].scene_label
                location_id = annotations[audio_filename][0].identifier
                if scene_label not in annotation_data:
                    annotation_data[scene_label] = {}
                if location_id not in annotation_data[scene_label]:
                    annotation_data[scene_label][location_id] = []
                annotation_data[scene_label][location_id].append(audio_filename)

            # Get event amounts
            event_amounts = {}
            for scene_label in list(annotation_data.keys()):
                if scene_label not in event_amounts:
                    event_amounts[scene_label] = {}
                for location_id in list(annotation_data[scene_label].keys()):
                    for audio_filename in annotation_data[scene_label][location_id]:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label, count in iteritems(current_event_amounts):
                            if event_label not in event_amounts[scene_label]:
                                event_amounts[scene_label][event_label] = 0
                            event_amounts[scene_label][event_label] += count

            for scene_label in list(annotation_data.keys()):
                # Optimize scene sets separately
                validation_set_candidates = []
                validation_set_MAE = []
                validation_set_event_amounts = []
                training_set_event_amounts = []
                for i in range(0, 1000):
                    location_ids = list(annotation_data[scene_label].keys())
                    random.shuffle(location_ids, random.random)

                    valid_percentage_index = int(np.ceil(valid_percentage * len(location_ids)))

                    current_validation_files = []
                    for loc_id in location_ids[0:valid_percentage_index]:
                        current_validation_files += annotation_data[scene_label][loc_id]

                    current_training_files = []
                    for loc_id in location_ids[valid_percentage_index:]:
                        current_training_files += annotation_data[scene_label][loc_id]

                    # event count in training set candidate
                    training_set_event_counts = np.zeros(len(event_amounts[scene_label]))
                    for audio_filename in current_training_files:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            if event_label in current_event_amounts:
                                training_set_event_counts[event_label_id] += current_event_amounts[event_label]

                    # Accept only sets which leave at least one example for training
                    if np.all(training_set_event_counts > 0):
                        # event counts in validation set candidate
                        validation_set_event_counts = np.zeros(len(event_amounts[scene_label]))
                        for audio_filename in current_validation_files:
                            current_event_amounts = annotations[audio_filename].event_stat_counts()

                            for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                                if event_label in current_event_amounts:
                                    validation_set_event_counts[event_label_id] += current_event_amounts[event_label]

                        # Accept only sets which have examples from each sound event class
                        if NotImplementedError.all(validation_set_event_counts > 0):
                            validation_amount = validation_set_event_counts / (validation_set_event_counts + training_set_event_counts)
                            validation_set_candidates.append(current_validation_files)
                            validation_set_MAE.append(mean_absolute_error(np.ones(len(validation_amount)) * valid_percentage, validation_amount))
                            validation_set_event_amounts.append(validation_set_event_counts)
                            training_set_event_amounts.append(training_set_event_counts)

                # Generate balance validation set
                # Selection done based on event counts (per scene class)
                # Target count specified percentage of training event count
                if validation_set_MAE:
                    best_set_id = np.argmin(validation_set_MAE)
                    validation_files += validation_set_candidates[best_set_id]

                    if self.show_extra_debug:
                        self.logger.debug('    Valid sets found [{sets}]'.format(sets=len(validation_set_MAE)))
                        self.logger.debug('    Best fitting set ID={id}, Error={error:4.2}%'.format(id=best_set_id, error=validation_set_MAE[best_set_id]*100))
                        self.logger.debug('    Validation event counts in respect of all data:')
                        event_amount_percentages = validation_set_event_amounts[best_set_id] / (validation_set_event_amounts[best_set_id] + training_set_event_amounts[best_set_id])
                        self.logger.debug('    {event:<20s} | {amount:10s} '.format(event='Event label', amount='Amount (%)'))
                        self.logger.debug('    {event:<20s} + {amount:10s} '.format(event='-' * 20, amount='-' * 20))

                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            self.logger.debug('    {event:<20s} | {amount:4.2f} '.format(
                                event=event_label,
                                amount=np.round(event_amount_percentages[event_label_id] * 100))
                            )

                else:
                    message = '{name}: Validation setup creation was not successful! Could not find a set with ' \
                              'examples for each event class in both training and validation.'.format(name=self.__class__.__name__)

                    self.logger.exception(message)
                    raise AssertionError(message)

        elif validation_type == 'generated_event_file_balanced':
            # Get event amounts
            event_amounts = {}
            for audio_filename in sorted(list(annotations.keys())):
                event_label = annotations[audio_filename][0].event_label
                if event_label not in event_amounts:
                    event_amounts[event_label] = []
                event_amounts[event_label].append(audio_filename)

            if self.show_extra_debug:
                self.logger.debug('    {event_label:<20s} | {amount:20s} '.format(
                    event_label='Event label',
                    amount='Files (%)')
                )

                self.logger.debug('    {event_label:<20s} + {amount:20s} '.format(
                    event_label='-' * 20,
                    amount='-' * 20)
                )

            def sorter(key):
                if not key:
                    return ""
                return key

            event_label_list = list(event_amounts.keys())
            event_label_list.sort(key=sorter)
            for event_label in event_label_list:
                files = np.array(list(event_amounts[event_label]))
                random.shuffle(files, random.random)
                valid_percentage_index = int(np.ceil(valid_percentage * len(files)))
                validation_files += files[0:valid_percentage_index].tolist()

                if self.show_extra_debug:
                    self.logger.debug('    {event_label:<20s} | {amount:4.2f} '.format(
                        event_label=event_label if event_label else '-',
                        amount=valid_percentage_index / float(len(files)) * 100.0)
                    )

            random.shuffle(validation_files, random.random)

        else:
            message = '{name}: Unknown validation_type [{type}].'.format(
                name=self.__class__.__name__,
                type=validation_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

        if self.show_extra_debug:
            self.logger.debug(' ')

        return sorted(validation_files)

    def learn(self, data, annotations, data_filenames=None, **kwargs):
        message = '{name}: Implement learn function.'.format(name=self.__class__.__name__)
        self.logger.exception(message)
        raise AssertionError(message)

class EventDetectorCRNN_Pytorch(EventDetector, PytorchMixin):
    def __init__(self, *args, **kwargs):
        super(EventDetectorCRNN_Pytorch, self).__init__(*args, **kwargs)

        self.method = 'pytorch_crnn'
        # length of one segment
        self.tau = self.learner_params.get_path('training.tau', 400)
        self.tau_step = self.learner_params.get_path('training.tau_step', 0.5)
        self.batch_size = self.learner_params.get_path('training.batch_size', 1)
        self.index = 0

    def CUDA(self, var):
        if torch.cuda.is_available():
            return var.cuda()
        else:
            return var

    def learn(self, data, annotations, data_filenames=None, validation_files=[], **kwargs):

        # Collect training files
        training_files = sorted(list(annotations.keys()))

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(
                    annotations=annotations,
                    validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                    valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                    seed=self.learner_params.get_path('validation.seed'),
                )
            elif self.learner_params.get_path('validation.setup_source') == 'dataset':
                if validation_files:
                    validation_files = sorted(list(set(validation_files).intersection(training_files)))
                else:
                    message = '{name}: No validation_files set'.format(name=self.__class__.__name__)
                    self.logger.exception(message)
                    raise ValueError(message)
            else:
                message = '{name}: Unknown validation [{mode}]'.format(name=self.__class__.__name__, mode=self.learner_params.get_path('validation.setup_source'))
                self.logger.exception(message)
                raise ValueError(message)

            training_files = sorted(list(set(training_files) - set(validation_files)))
        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(name=self.__class__.__name__)
            self.logger.exception(message)
            raise ValueError(message)

        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        # Process data
        X_training = self.prepare_data(data=data, files=training_files)
        Y_training = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)
        self.feature_dim = len(X_training[0])

        if self.show_extra_debug:
            self.logger.debug('  Training Positive frames \t\t[{examples:d}]'.format(examples=int(np.sum((Y_training == 1)))))
            self.logger.debug('  Training Negative frames \t\t[{examples:d}]'.format(examples=int(np.sum((Y_training == 0)))))

        # Process validation data
        if validation_files:
            X_validation = self.prepare_data(data=data, files=validation_files)
            Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)
            if self.show_extra_debug:
                self.logger.debug('  Validation Positive frames \t[{examples:d}]'.format(examples=int(np.sum(Y_validation))))
                self.logger.debug('  Validation Negative frames \t[{examples:d}]'.format(examples=len(Y_validation)-int(np.sum(Y_validation))))

        # convert dataset to pytorch style
        training_dataset = SegmentedDataset(X_training, Y_training, self.tau, self.tau_step, smooth=False, trailing_label=True)
        self.data_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        validation_dataset = SegmentedDataset(X_validation, Y_validation, self.tau, self.tau_step, smooth=False, trailing_label=True)
        self.validation_data_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        # create model
        self.model = self.CUDA(CRNN_Reverse_Pytorch(input_dim=self._get_input_size(data=data), tau=self.tau, feature_dim=self.feature_dim))
        self.model.apply(kaiming_init)

        # create placeholder
        self.data_x = self.CUDA(Variable(torch.FloatTensor(self.batch_size, self.tau, self.feature_dim)))
        self.label_y = self.CUDA(Variable(torch.FloatTensor(self.batch_size, self.tau, 1)))

        # create optimizer
        self.optimzer = optim.Adam([{'params': self.model.parameters(), 'lr': self.learner_params.get_path('training.learning_rate', 0.001)}])

        self.max_iter = int(training_dataset.__len__())
        self.epoch = self.learner_params.get_path('training.epochs', 1)
        self.iter_per_epoch = int(self.max_iter/self.batch_size)

        if self.show_extra_debug:
            self.logger.debug('  Feature dim \t[{vector:d}]'.format(vector=self._get_input_size(data=data)))
            self.logger.debug('  Learning Rate \t[{lr:f}]'.format(lr=self.learner_params.get_path('training.learning_rate', 0.001)))
            self.logger.debug('  Tau \t\t[{tau:f}]'.format(tau=self.tau))
            self.logger.debug('  Tau Step \t\t[{tau:f}]'.format(tau=self.tau_step))
            self.logger.debug('  ------------')
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=self.max_iter))
            self.logger.debug('  Validation items \t[{examples:d}]'.format(examples=int(validation_dataset.__len__())))
            self.logger.debug('  Batch size \t[{batch:d}]'.format(batch=self.learner_params.get_path('training.batch_size', 1)))
            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(epoch=self.learner_params.get_path('training.epochs', 1)))
            print('\n')

        self.pbar = tqdm(desc="           {0: >15s}".format('Training '), total=self.iter_per_epoch*self.epoch)

        print_loss = []
        lowest_validation_loss = 9999
        best_epoch = 0
        for e in range(self.epoch):
            self.global_iter = 0
            self.model.train()
            for x, y in self.data_loader:
                self.batch_size = x.size(0)

                self.data_x.data.resize_(self.batch_size, self.tau, self.feature_dim)
                self.label_y.data.resize_(self.batch_size, self.tau, 1)
                self.global_iter += 1
                
                self.data_x.data.copy_(x)
                self.label_y.data.copy_(y)
                
                output = self.model(self.data_x)
                loss = F.binary_cross_entropy(output, self.label_y)

                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()
                self.pbar.update(1)
                print_loss.append(loss.data.cpu().numpy())
                
                if self.global_iter % 10 == 0:
                    self.pbar.write('Epoch: [{}\{}], Iteration: [{}/{}], Binary Cross Entropy Loss: {}'.format(
                        e, self.epoch, self.global_iter, self.iter_per_epoch, np.mean(print_loss))
                    )
                    print_loss = []

            # do validation
            validation_loss = self.validation()
            #validation_loss = self.target_task_validation()

            if lowest_validation_loss >= validation_loss:
                best_epoch = e
                lowest_validation_loss = validation_loss
                best_model_states = copy.deepcopy(self.model.state_dict())
                self.pbar.write('Save best model !')

            self.pbar.write('Epoch: [{}\{}], Validation Loss: {}, Lowest validation loss: {} (Epoch: {})'.format(
                e, 
                self.epoch, 
                validation_loss, 
                lowest_validation_loss, 
                best_epoch)
            )

        self.pbar.close()

        # restore the best model of validation dataset
        self.model.load_state_dict(best_model_states)

    def predict(self, feature_data):
        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]

        '''
        np.save('/home/dingwh/mel/'+str(self.index)+'_mel.npy', feature_data)
        self.index += 1
        '''

        # set the mode to evaluation, forbid Batch normalization and dropout
        self.model.eval()

        feature_pytorch = self.CUDA(Variable(torch.FloatTensor(1, len(feature_data), len(feature_data[0]))))
        feature_pytorch.data.copy_(torch.from_numpy(feature_data).unsqueeze(0))

        return self.model(feature_pytorch).data.cpu().numpy().T

    def validation(self):
        self.model.eval()
        validation_loss = []

        for x, y in self.validation_data_loader:
            self.batch_size = x.size(0)

            self.data_x.data.resize_(self.batch_size, self.tau, self.feature_dim)
            self.label_y.data.resize_(self.batch_size, self.tau, 1)
            self.data_x.data.copy_(x)
            self.label_y.data.copy_(y)

            output = self.model(self.data_x)
            validation_loss.append(F.binary_cross_entropy(output, self.label_y).data.cpu().numpy())

        return np.mean(validation_loss)
    
    def target_task_validation(self):
        self.model.eval()
        validation_loss = []

        for x, y in self.validation_data_loader:
            self.batch_size = x.size(0)

            self.data_x.data.resize_(self.batch_size, 1501, self.feature_dim)
            self.label_y.data.resize_(self.batch_size, 1501, 1)
            self.data_x.data.copy_(x)
            self.label_y.data.copy_(y)

            output = self.model(self.data_x)
            validation_loss.append(F.binary_cross_entropy(output, self.label_y).data.cpu().numpy())

        return np.mean(validation_loss)

