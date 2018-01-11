JobTitle2Vec
=======

This repository shows how to use Word2Vec to build a JobTitle2Vec model. Test

## Getting Started

```sh
# install pipenv if you don't have it
pip install pipenv

# install all dependency packages
pipenv install

# to train the model file
pipenv run python train.py

# to test using the JobTitle2Vec model
pipenv run python train.py
```

## API

#### constructor(model_file)

Provide a model file when create the model

#### load(model_file)

Load from a model file

#### get_vector(job_title)

Get the vector representation of the job title

#### similarity(x, y)

Get the similarity score (range from 0 to 1) for 2 job titles x and y
