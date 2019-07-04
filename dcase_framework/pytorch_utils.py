import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch

import logging
import numpy
import copy

from .data import DataBuffer

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def CUDA(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CRNN_Reverse_Pytorch(nn.Module):
    def __init__(self, input_dim, tau, feature_dim):
        super(CRNN_Reverse_Pytorch, self).__init__()
        self.input_dim = input_dim
        self.channels = 1
        self.hidden_size = 64
        self.rnn_input_size = 128
        self.num_layers = 2
        self.tau = tau
        self.feature_dim = feature_dim
        self.conv_kernel_size = 32
        self.p = 0.3
        
        self.branch1x1_1 = BasicConv2d(1, 32, kernel_size=1) 
        
        self.branch3x3_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(32,64, kernel_size=(1, 3), padding=(0, 1)) 
        self.branch3x3_3 = BasicConv2d(64,32, kernel_size=(3, 1), padding=(1, 0)) 
        
        self.branch5x5_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32,64, kernel_size=(1, 5), padding=(0, 2))      
        self.branch5x5_3 = BasicConv2d(64,32, kernel_size=(5, 1), padding=(2, 0)) 
              
        self.branch7x7_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(32,64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(64,32, kernel_size=(7, 1), padding=(3, 0))       


        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 128), stride=1)#97

        # rnn-gru layer
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=self.p
        )

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.p),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*2, batch_size, hidden_size)))

    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size = x.size(0)
       
        branch1x1 = self.branch1x1_1(x)       
        

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
               
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        outputs = (branch1x1,branch3x3,branch5x5,branch7x7)
        feature = torch.cat(outputs, 1)
        output = self.max_pool(feature).squeeze(-1).permute(0,2,1)
        h_state = self.init_hidden(batch_size, self.hidden_size)
        self.rnn.flatten_parameters()
        output, h_state = self.rnn(output, h_state)
        output = self.fc(output[:, :, self.hidden_size:]).squeeze(0)
        return output

class SegmentedDataset(Dataset):
    def __init__(self, training_X, training_Y, tau, tau_step, smooth=False, trailing_label=False):
        self.tau = tau
        self.train_data = []
        self.train_label = []
        self.tau_step = tau_step
        self.training_X_inserted = []
        self.training_Y_inserted = []
        
        if smooth:
            # label smoothing
            smooth_label_1 = 0.3
            smooth_label_2 = 0.5
            smooth_label_3 = 0.7

            for i in range(len(training_Y)-1):
                self.training_X_inserted.append(training_X[i,:])
                self.training_Y_inserted.append(training_Y[i])

                # onset or offset
                if (training_Y[i,:] == 0 and training_Y[i+1,:] == 1) or training_Y[i,:] == 1 and training_Y[i+1,:] == 0:
                    
                    smooth_1 = smooth_label_1*training_X[i+1,:]+(1-smooth_label_1)*training_X[i,:]
                    smooth_2 = smooth_label_2*training_X[i+1,:]+(1-smooth_label_2)*training_X[i,:]
                    smooth_3 = smooth_label_3*training_X[i+1,:]+(1-smooth_label_3)*training_X[i,:]

                    # insert three frames
                    self.training_X_inserted.append(smooth_1)
                    self.training_X_inserted.append(smooth_2)
                    self.training_X_inserted.append(smooth_3)
                    self.training_Y_inserted.append([smooth_label_1])
                    self.training_Y_inserted.append([smooth_label_2])
                    self.training_Y_inserted.append([smooth_label_3])

            # add the last frame
            self.training_X_inserted.append(training_X[i,:])
            self.training_Y_inserted.append(training_Y[i,:])

            print('After smoothing, length of traning data: {}'.format(len(self.training_X_inserted)))
            print('After smoothing, length of traning label:: {}'.format(len(self.training_Y_inserted)))

            # convert to numpy style
            self.training_X_inserted = np.array(self.training_X_inserted)
            self.training_Y_inserted = np.array(self.training_Y_inserted)
        else:
            self.training_X_inserted = training_X
            self.training_Y_inserted = training_Y

        # use trailing label in reverse sequence
        if trailing_label:
            for i in range(len(self.training_Y_inserted)-1):
                if self.training_Y_inserted[i,:] == 1 and self.training_Y_inserted[i+1,:] == 0:
                    self.training_Y_inserted[i,:]   = 0.1
                    self.training_Y_inserted[i-1,:] = 0.2
                    self.training_Y_inserted[i-2,:] = 0.3
                    self.training_Y_inserted[i-3,:] = 0.4
                    self.training_Y_inserted[i-4,:] = 0.5
                    self.training_Y_inserted[i-5,:] = 0.6
                    self.training_Y_inserted[i-6,:] = 0.7
                    self.training_Y_inserted[i-7,:] = 0.8
                    self.training_Y_inserted[i-8,:] = 0.9
            print('\tFinished trailing label')
        # separate the traning data
        seg = 0
        while(1):
            start_ = int(seg*self.tau_step*self.tau)
            end_ = int((self.tau_step*seg+1)*self.tau)
            if end_ > len(self.training_X_inserted):
                break
            self.train_data.append(self.training_X_inserted[start_:end_,:])
            self.train_label.append(self.training_Y_inserted[start_:end_,:])
            seg += 1

        # convert to numpy style
        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)

        self.indices = range(len(self))

    def __getitem__(self, index):
        data = self.train_data[index,:,:]
        label = self.train_label[index,:,:]

        return data, label

    def __len__(self):
        return len(self.train_data)


class PytorchMixin(object):
    def prepare_data(self, data, files, processor='default'):
        """
            Concatenate feature data into one feature matrix

            Parameters
            ----------
            data : dict of FeatureContainers
                Feature data
            files : list of str
                List of filenames
            processor : str ('default', 'training')
                Data processor selector
                Default value 'default'

            Returns
            -------
            numpy.ndarray
                Features concatenated
        """

        if self.learner_params.get_path('input_sequencer.enable'):
            processed_data = []
            for item in files:
                if processor == 'training':
                    processed_data.append(self.data_processor_training.process_data(data=data[item].feat[0]))
                else:
                    processed_data.append(self.data_processor.process_data(data=data[item].feat[0]))
            return np.concatenate(processed_data)
        else:
            return np.vstack([data[x].feat[0] for x in files])
    
    def prepare_activity(self, activity_matrix_dict, files, processor='default'):
        """
            Concatenate activity matrices into one activity matrix

            Parameters
            ----------
            activity_matrix_dict : dict of binary matrices
                Meta data
            files : list of str
                List of filenames
            processor : str ('default', 'training')
                Data processor selector
                Default value 'default'
            Returns
            -------
            numpy.ndarray
                Activity matrix
        """

        if self.learner_params.get_path('input_sequencer.enable'):
            processed_activity = []
            for item in files:
                if processor == 'training':
                    processed_activity.append(self.data_processor_training.process_activity_data(activity_data=activity_matrix_dict[item]))
                else:
                    processed_activity.append(self.data_processor.process_activity_data(activity_data=activity_matrix_dict[item]))
            return np.concatenate(processed_activity)
        else:
            return np.vstack([activity_matrix_dict[x] for x in files])

    def create_external_metric_evaluators(self):
        """
            Create external metric evaluators
        """

        # Initialize external metrics
        external_metric_evaluators = collections.OrderedDict()
        if self.learner_params.get_path('training.epoch_processing.enable'):
            if self.learner_params.get_path('validation.enable') and self.learner_params.get_path(
                    'training.epoch_processing.external_metrics.enable'):
                import sed_eval

                for metric in self.learner_params.get_path('training.epoch_processing.external_metrics.metrics'):
                    # Current metric info
                    current_metric_evaluator = metric.get('evaluator')
                    current_metric_name = metric.get('name')
                    current_metric_params = metric.get('parameters', {})
                    current_metric_label = metric.get('label', current_metric_name.split('.')[-1])

                    # Initialize sed_eval evaluators
                    if current_metric_evaluator == 'sed_eval.scene':
                        evaluator = sed_eval.scene.SceneClassificationMetrics(scene_labels=self.class_labels, **current_metric_params)
                    elif (current_metric_evaluator == 'sed_eval.segment_based' or current_metric_evaluator == 'sed_eval.sound_event.segment_based'):
                        evaluator = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=self.class_labels, **current_metric_params)
                    elif (current_metric_evaluator == 'sed_eval.event_based' or current_metric_evaluator == 'sed_eval.sound_event.event_based'):
                        evaluator = sed_eval.sound_event.EventBasedMetrics(event_label_list=self.class_labels, **current_metric_params)
                    else:
                        message = '{name}: Unknown target metric [{metric}].'.format(name=self.__class__.__name__, metric=current_metric_name)
                        self.logger.exception(message)
                        raise AssertionError(message)

                    # Check evaluator API
                    if (not hasattr(evaluator, 'reset') or not hasattr(evaluator, 'evaluate') or not hasattr(evaluator, 'results')):
                        if current_metric_evaluator.startswith('sed_eval'):
                            message = '{name}: wrong version of sed_eval for [{current_metric_evaluator}::{current_metric_name}], update sed_eval to latest version'.format(
                                name=self.__class__.__name__,
                                current_metric_evaluator=current_metric_evaluator,
                                current_metric_name=current_metric_name
                            )

                            self.logger.exception(message)
                            raise ValueError(message)
                        else:
                            message = '{name}: Evaluator has invalid API [{current_metric_evaluator}::{current_metric_name}]'.format(
                                name=self.__class__.__name__,
                                current_metric_evaluator=current_metric_evaluator,
                                current_metric_name=current_metric_name
                            )

                            self.logger.exception(message)
                            raise ValueError(message)

                    # Form unique name for metric, to allow multiple similar metrics with different parameters
                    metric_id = get_parameter_hash(metric)

                    # Metric data container
                    metric_data = {
                        'evaluator_name': current_metric_evaluator,
                        'name': current_metric_name,
                        'params': current_metric_params,
                        'label': current_metric_label,
                        'path': current_metric_name,
                        'evaluator': evaluator,
                    }
                    external_metric_evaluators[metric_id] = metric_data

        return external_metric_evaluators

class BaseDataGenerator(object):
    """
        Base class for data generator.
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor

            Parameters
            ----------
            files : list

            data_filenames : dict

            annotations : dict

            class_labels : list of str

            hop_length_seconds : float
                Default value 0.2

            shuffle : bool
                Default value True

            batch_size : int
                Default value 30

            buffer_size : int
                Default value 256
        """

        self.method = 'base_generator'

        # Data
        self.item_list = copy.copy(kwargs.get('files', []))
        self.data_filenames = kwargs.get('data_filenames', {})
        self.annotations = kwargs.get('annotations', {})

        # Activity matrix
        self.class_labels = kwargs.get('class_labels', [])
        self.hop_length_seconds = kwargs.get('hop_length_seconds', 0.2)

        self.shuffle = kwargs.get('shuffle', True)
        self.batch_size = kwargs.get('batch_size', 64)
        self.buffer_size = kwargs.get('buffer_size', 256)

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        # Internal state variables
        self.batch_index = 0
        self.item_index = 0
        self.data_position = 0

        # Initialize data buffer
        self.data_buffer = DataBuffer(size=self.buffer_size)

        if self.buffer_size >= len(self.item_list):
            # Fill data buffer at initialization if it fits fully to the buffer
            for current_item in self.item_list:
                self.process_item(item=current_item)
                if self.data_buffer.full():
                    break

        self._data_size = None
        self._input_size = None

    @property
    def steps_count(self):
        """
            Number of batches in one epoch
        """
        num_batches = int(numpy.ceil(self.data_size / float(self.batch_size)))
        if num_batches > 0:
            return num_batches
        else:
            return 1

    @property
    def input_size(self):
        """
            Length of input feature vector
        """
        if self._input_size is None:
            # Load first item
            first_item = list(self.data_filenames.keys())[0]
            self.process_item(item=first_item)
            # Get Feature vector length
            self._input_size = self.data_buffer.get(key=first_item)[0].shape[-1]

        return self._input_size

    @property
    def data_size(self):
        """
            Total data amount
        """
        if self._data_size is None:
            self._data_size = 0
            for current_item in self.item_list:
                self.process_item(item=current_item)
                data, meta = self.data_buffer.get(key=current_item)
                # Accumulate feature matrix length
                self._data_size += data.shape[0]

        return self._data_size

    def info(self):
        """
            Information logging
        """
        info = [
            '  Generator',
            '    Shuffle \t[{shuffle}]'.format(shuffle='True' if self.shuffle else 'False'),
            '    Epoch size\t[{steps:d} batches]'.format(steps=self.steps_count),
            '    Buffer size \t[{buffer_size:d} files]'.format(buffer_size=self.buffer_size),
            ' '
        ]
        return info

    def process_item(self, item):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass


class FeatureGenerator(BaseDataGenerator):
    """
        Feature data generator
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor

            Parameters
            ----------
            files : list of str
                List of active item identifies, usually filenames

            data_filenames : dict of dicts
                Data structure keyed with item identifiers (defined with files parameter), data dict feature extractor
                labels as keys and values the filename on disk.

            annotations : dict of MetaDataContainers or MetaDataItems
                Annotations for all items keyed with item identifiers

            class_labels : list of str
                Class labels in a list

            hop_length_seconds : float
                Analysis frame hop length in seconds
                Default value 0.2

            shuffle : bool
                Shuffle data before each epoch
                Default value True

            batch_size : int
                Batch size to generate
                Default value 64

            buffer_size : int
                Internal item buffer size, set large enough for smaller dataset to avoid loading
                Default value 256

            data_processor : class
                Data processor class used to process load features

            data_refresh_on_each_epoch : bool
                Internal data buffer reset at the start of each epoch
                Default value False

            label_mode : str ('event', 'scene')
                Activity matrix forming mode.
                Default value "event"
        """
        self.data_processor = kwargs.get('data_processor')
        self.data_refresh_on_each_epoch = kwargs.get('data_refresh_on_each_epoch', False)
        self.label_mode = kwargs.get('label_mode', 'event')

        super(FeatureGenerator, self).__init__(*args, **kwargs)

        self.method = 'feature'

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        if self.label_mode not in ['event', 'scene']:
            message = '{name}: Label mode unknown [{label_mode}]'.format(name=self.__class__.__name__, metric=self.label_mode)
            self.logger.exception(message)
            raise ValueError(message)

    def process_item(self, item):
        if not self.data_buffer.key_exists(key=item):
            current_data, current_length = self.data_processor.load(feature_filename_dict=self.data_filenames[item])
            current_activity_matrix = self.get_activity_matrix(annotation=self.annotations[item], data_length=current_length)
            self.data_buffer.set(key=item, data=current_data, meta=current_activity_matrix)

    def on_epoch_start(self):
        self.batch_index = 0
        if self.shuffle:
            # Shuffle item list order
            numpy.random.shuffle(self.item_list)

        if self.data_refresh_on_each_epoch:
            # Force reload of data
            self.data_buffer.clear()

    def generator(self):
        """
            Generator method

            Returns
            -------
            ndarray
                data batches
        """
        while True:
            # Start of epoch
            self.on_epoch_start()

            batch_buffer_data = []
            batch_buffer_meta = []

            # Go through items
            for item in self.item_list:
                # Load item data into buffer
                self.process_item(item=item)

                # Fetch item from buffer
                data, meta = self.data_buffer.get(key=item)

                # Data indexing
                data_ids = numpy.arange(data.shape[0])

                # Shuffle data order
                if self.shuffle:
                    numpy.random.shuffle(data_ids)

                for data_id in data_ids:
                    if len(batch_buffer_data) == self.batch_size:
                        # Batch buffer full, yield data
                        yield (numpy.concatenate(numpy.expand_dims(batch_buffer_data, axis=0)), numpy.concatenate(numpy.expand_dims(batch_buffer_meta, axis=0)))

                        # Empty batch buffers
                        batch_buffer_data = []
                        batch_buffer_meta = []

                        # Increase batch counter
                        self.batch_index += 1

                    # Collect data fro the batch
                    batch_buffer_data.append(data[data_id])
                    batch_buffer_meta.append(meta[data_id])

            if len(batch_buffer_data):
                # Last batch, usually not full
                yield (numpy.concatenate(numpy.expand_dims(batch_buffer_data, axis=0)), numpy.concatenate(numpy.expand_dims(batch_buffer_meta, axis=0)))
                # Increase batch counter
                self.batch_index += 1

            # End of epoch
            self.on_epoch_end()

    def get_activity_matrix(self, annotation, data_length):
        """
            Convert annotation into activity matrix and run it through data processor.
        """
        event_roll = None
        if self.label_mode == 'event':
            # Event activity, event onset and offset specified
            event_roll = EventRoll(metadata_container=annotation, label_list=self.class_labels, time_resolution=self.hop_length_seconds)
            event_roll = event_roll.pad(length=data_length)
        elif self.label_mode == 'scene':
            # Scene activity, one-hot activity throughout whole file
            pos = self.class_labels.index(annotation.scene_label)
            event_roll = numpy.zeros((data_length, len(self.class_labels)))
            event_roll[:, pos] = 1

        if event_roll is not None:
            return self.data_processor.process_activity_data(activity_data=event_roll)
        else:
            return None
