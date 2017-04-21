# ML Project Template

Template for a new Python ML project.

## Dev environment setup

Python is the main language used in this codebase.
We strongly encourage the use of Python [virtual environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

    virtualenv venv
    source venv/bin/activate

After which, you can install the required Python modules via

    pip install -r requirements.txt

## Data

All processed data for training, evaluation, etc can be found in the [`data`](data/) folder.
See the [README](data/README.md) for information about the different datasets available.

## Model training and evaluation

### Step 1: Convert instances to sparse vectors (Featurization)

The [`project.featurize`](project/featurize.py) script will perform the necessary steps to convert JSON instance files into featurized `numpy` arrays.

```
(venv) $ python -m project.featurize -h
usage: featurize.py [-h] [-i [<instances> [<instances> ...]]]
                    [-o <features_uri>] [-s <settings_uri>] [--n-jobs <N>]
                    [--log-level <log_level>]
                    (-f <featurizer> | -t <featurizer> | -z <featurizer> [<featurizer> ...] | -x <features_uri> [<features_uri> ...] | -v <featurizer> <features_uri>)
                    [<featurizer_type>]

Featurize instances for ML classification.

positional arguments:
  <featurizer_type>     Name of featurizer model to use.

optional arguments:
  -h, --help            show this help message and exit
  -i [<instances> [<instances> ...]], --instances [<instances> [<instances> ...]]
                        List of instance files to featurize.
  -o <features_uri>, --output <features_uri>
                        Save featurized instances here.
  -s <settings_uri>, --settings <settings_uri>
                        Settings file to configure models.
  --n-jobs <N>          No. of processes to use during featurization.
  --log-level <log_level>
                        Set log level of logger.
  -f <featurizer>, --fit <featurizer>
                        Fit instances and save featurizer model file here.
  -t <featurizer>, --featurize <featurizer>
                        Use this featurizer to transform instances.
  -z <featurizer> [<featurizer> ...], --featurizer-info <featurizer> [<featurizer> ...]
                        Display information about featurizer model.
  -x <features_uri> [<features_uri> ...], --features-info <features_uri> [<features_uri> ...]
                        Display information about featurized instance file.
  -v <featurizer> <features_uri>, --verify <featurizer> <features_uri>
                        Verify that the featurized instance file came from the
                        same featurizer model.
```

Example:

    python -m project.featurize ExampleFeaturizer -i ./data/train.json.gz --fit models/development.featurizer.gz -o data/development.features.npz

You can set parameters for the featurizer through the settings file directly (or use the default).
We store existing settings file in [`settings/`](settings/) using the naming convention of `<environment>.settings.yaml`, where settings for multiple apps are stored in a single YAML file.

The following featurizers are available:

- `HelloWorld`: Description

### Step 2: Train/Evaluate model

The [`project.classify`](project/classify.py) script will perform the necessary steps fit/evaluate/use a model from featurized instances.

```
(venv) $ python -m project.classify -h
usage: classify.py [-h] [--log-level <log_level>] [-s <settings_file>]
                   [-c <classifier_file>]
                   <mode> ...

Classify instances for AOLs and Issues classifier.

optional arguments:
  -h, --help            show this help message and exit
  --log-level <log_level>
                        Set log level of logger.
  -s <settings_file>, --settings <settings_file>
                        Settings file to configure models.
  -c <classifier_file>, --classifier-info <classifier_file>
                        Display information about classifier.

Different classifier modes for fitting, evaluating, and prediction.:
  <mode>
    fit                 Fit an AOLs and Issues classifier.
    evaluate            Evaluate an AOLs and Issues classifier.
    predict             Predict using an AOLs and Issues classifier.
```

