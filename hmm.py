# -*- coding: utf-8 -*-

def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob[item] / prob_sum
        else:
            for item in item_set:
                result[item] = 0

    return result

def _normalize_prob_two_dim(prob, item_set1, item_set2):
    result = {}
    if prob is None:
        for item in item_set1:
            result[item] = _normalize_prob(None, item_set2)
    else:
        for item in item_set1:
            if item in prob:
                result[item] = _normalize_prob(prob[item], item_set2)
            else:
                result[item] = _normalize_prob(None, item_set2)

    return result

class Model(object):
    def __init__(self, states, symbols, start_prob=None, trans_prob=None, emit_prob=None):
        self._states = set(states)
        self._symbols = set(symbols)
        self._start_prob = _normalize_prob(start_prob, self._states)
        self._trans_prob = _normalize_prob_two_dim(trans_prob, self._states, self._states)
        self._emit_prob = _normalize_prob_two_dim(emit_prob, self._states, self._symbols)

    def states(self):
        return self._states[:]

    def states_number(self):
        return len(self._states)

    def symbols(self):
        return self._symbols[:]

    def symbols_number(self):
        return len(self._symbols)

    def start_prob(self, state):
        if state not in self._states:
            return 0
        return self._start_prob[state]

    def trans_prob(self, state_from, state_to):
        if state_from not in self._states or state_to not in self._states:
            return 0
        return self._trans_prob[state_from][state_to]

    def emit_prob(self, state, symbol):
        if state not in self._states or symbol not in self._symbols:
            return 0
        return self._emit_prob[state][symbol]

