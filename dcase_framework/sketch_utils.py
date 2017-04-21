import importlib
import copy
from dcase_framework.containers import DottedDict

def create_model(learner_params, input_shape):
    from keras.models import Sequential
    model = Sequential()
    model_params = copy.deepcopy(learner_params.get_path('model.config'))

    if 'conv_kernel' in learner_params['model']:
        kernel_params = learner_params['model']['conv_kernel']
        kernel = getattr(importlib.import_module("keras.initializers"), kernel_params['class'])
        kernel = kernel(**kernel_params['config'])
        set_kernel = True

    for layer_id, layer_setup in enumerate(model_params):
        layer_setup = DottedDict(layer_setup)
        try:
            LayerClass = getattr(importlib.import_module("keras.layers"), layer_setup['class_name'])
        except AttributeError:
            message = '{name}: Invalid Keras layer type [{type}].'.format(
                name=__class__.__name__,
                type=layer_setup['class_name']
            )
            raise AttributeError(message)

        if 'config' not in layer_setup:
            layer_setup['config'] = {}

        # Set layer input
        if layer_id == 0 and layer_setup.get_path('config.input_shape') is None:
            # Set input layer dimension for the first layer if not set
            if layer_setup.get('class_name') == 'Dropout':
                layer_setup['config']['input_shape'] = (input_shape,)
            else:
                layer_setup['config']['input_shape'] = input_shape

        elif layer_setup.get_path('config.input_dim') == 'FEATURE_VECTOR_LENGTH':
            layer_setup['config']['input_dim'] = input_shape

        # Set layer output
        if layer_setup.get_path('config.units') == 'CLASS_COUNT':
            layer_setup['config']['units'] = len(class_labels)

        # Set kernel initializer
        if 'kernel_initializer' in layer_setup.get('config') and set_kernel:
            layer_setup.get('config')['kernel_initializer'] = kernel

        if layer_setup.get('config'):
            model.add(LayerClass(**dict(layer_setup.get('config'))))
        else:
            model.add(LayerClass())

    try:
        OptimizerClass = getattr(importlib.import_module("keras.optimizers"),
                                 learner_params.get_path('model.optimizer.type')
                                 )

    except AttributeError:
        message = '{name}: Invalid Keras optimizer type [{type}].'.format(
            name=__class__.__name__,
            type=learner_params.get_path('model.optimizer.type')
        )
        raise AttributeError(message)

    model.compile(
        loss=learner_params.get_path('model.loss'),
        optimizer=OptimizerClass(**dict(learner_params.get_path('model.optimizer.parameters', {}))),
        metrics=learner_params.get_path('model.metrics')
    )

    return model




