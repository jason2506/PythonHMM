# PythonHMM

**PythonHMM** is a python implementation of the [Hidden Markov Model](http://en.wikipedia.org/wiki/Hidden_Markov_model).

## Usage

To use **PythonHMM**, you must import the `hmm` module.

    import hmm

Then, you can create an instance of `Model` by passing the states, symbols, and (optional) probability matrices.

    states = ('rainy', 'sunny')
    symbols = ('walk', 'shop', 'clean')

    start_prob = {
        'rainy' : 0.5,
        'sunny' : 0.5
    }

    trans_prob = {
        'rainy': { 'rainy' : 0.7, 'sunny' : 0.3 },
        'sunny': { 'rainy' : 0.4, 'sunny' : 0.6 }
    }

    emit_prob = {
        'rainy': { 'walk' : 0.1, 'shop' : 0.4, 'clean' : 0.5 },
        'sunny': { 'walk' : 0.6, 'shop' : 0.3, 'clean' : 0.1 }
    }

    model = hmm.Model(states, symbols, start_prob, trans_prob, emit_prob)

Now, you can evaluate and decode the given sequence:

    sequence = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk', 'walk', 'clean']

    print model.evaluate(sequence)
    print model.decode(sequence)

You can also using the given sequences (a list of *(state list, symbol list)* pair) to train a model:

    sequences = [
        (state_list1, symbol_list1),
        (state_list2, symbol_list2),
        ...
        (state_listN, symbol_listN),
    ]

    model = hmm.train(sequences)

The `train` function also has two optional arguments, `delta` and `smoothing`.

The `delta` argument (which is defaults to 0.0001) specifies that the learning algorithm will stop when the difference of the log-likelihood between two consecutive iterations is less than `delta`.

The `smoothing` argument (which is defaults to 0) is the smoothing parameter of the [additive smoothing](http://en.wikipedia.org/wiki/Additive_smoothing) to avoid zero probability.

## License

This project is [BSD-licensed](http://www.opensource.org/licenses/BSD-3-Clause). See LICENSE file for more detail.
