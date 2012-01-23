# PythonHMM

## Overview

**PythonHMM** is the implementation of the [Hidden Markov Model](http://en.wikipedia.org/wiki/Hidden_Markov_model).

## Usage

    from hmm import Model

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

    sequence = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk', 'walk', 'clean']
    model = Model(states, symbols, start_prob, trans_prob, emit_prob)

    print model.evaluate(sequence)

## License

This project is [BSD-licensed](http://www.opensource.org/licenses/BSD-3-Clause). See LICENSE file for more detail.
