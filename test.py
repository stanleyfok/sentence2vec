from lib.sentence2vec import Sentence2Vec

model = Sentence2Vec('./data/job_titles.model')

# turn job title to vector
print(model.get_vector('Uber Driver Partner'))

# not similar job
print(model.similarity('Uber Driver Partner',
                       'Carpenter/ Modular  building installer'))

# a bit similar job
print(model.similarity('Temporary Barista 30 hours per week',
                       'Waitress / Waiter Part-Timer'))

# similar job
print(model.similarity('Sandwich maker / All rounder',
                       'Cafe all rounder and Sandwich Hand'))
