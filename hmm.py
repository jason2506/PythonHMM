# -*- coding: utf-8 -*-

def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
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
            result[item] = _normalize_prob(prob.get(item), item_set2)

    return result

class Model(object):
    def __init__(self, states, symbols, start_prob=None, trans_prob=None, emit_prob=None):
        self._states = set(states)
        self._symbols = set(symbols)
        self._start_prob = _normalize_prob(start_prob, self._states)
        self._trans_prob = _normalize_prob_two_dim(trans_prob, self._states, self._states)
        self._emit_prob = _normalize_prob_two_dim(emit_prob, self._states, self._symbols)

    def states(self):
        return set(self._states)

    def states_number(self):
        return len(self._states)

    def symbols(self):
        return set(self._symbols)

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

    def _forward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        alpha = [{}]
        for state in self._states:
            alpha[0][state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

        for index in xrange(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                prob = 0
                for state_from in self._states:
                    prob += alpha[index - 1][state_from] * \
                        self.trans_prob(state_from, state_to)
                alpha[index][state_to] = prob * self.emit_prob(state_to, sequence[index])

        return alpha

    def _backward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        beta = [{}]
        for state in self._states:
            beta[0][state] = 1

        for index in xrange(sequence_length - 1, 0, -1):
            beta.insert(0, {})
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    prob += beta[1][state_to] * \
                        self.trans_prob(state_from, state_to) * \
                        self.emit_prob(state_to, sequence[index])
                beta[0][state_from] = prob

        return beta

    def evaluate(self, sequence):
        length = len(sequence)
        if length == 0:
            return 0

        prob = 0
        alpha = self._forward(sequence)
        for state in alpha[length - 1]:
            prob += alpha[length - 1][state]

        return prob

    def decode(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        delta = {}
        for state in self._states:
            delta[state] = self.start_prob(state) * self.emit_prob(state, sequence[0])

        pre = []
        for index in xrange(1, sequence_length):
            delta_bar = {}
            pre_state = {}
            for state_to in self._states:
                max_prob = 0
                max_state = None
                for state_from in self._states:
                    prob = delta[state_from] * self.trans_prob(state_from, state_to)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from
                delta_bar[state_to] = max_prob * self.emit_prob(state_to, sequence[index])
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0
        for state in self._states:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state

        if max_state is None:
            return []

        result = [max_state]
        for index in xrange(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)

        return result

    def learn(self, sequence):
        length = len(sequence)
        alpha = self._forward(sequence)
        beta = self._backward(sequence)

        gamma = []
        for index in xrange(length):
            prob_sum = 0
            gamma.append({})
            for state in self._states:
                prob = alpha[index][state] * beta[index][state]
                gamma[index][state] = prob
                prob_sum += prob

            if prob_sum == 0:
                continue

            for state in self._states:
                gamma[index][state] /= prob_sum

        xi = []
        for index in xrange(length - 1):
            prob_sum = 0
            xi.append({})
            for state_from in self._states:
                xi[index][state_from] = {}
                for state_to in self._states:
                    prob = alpha[index][state_from] * beta[index + 1][state_to] * \
                        self._trans_prob[state_from][state_to] * \
                        self._emit_prob[state_to][sequence[index]]
                    xi[index][state_from][state_to] = prob
                    prob_sum += prob

            if prob_sum == 0:
                continue

            for state_from in self._states:
                for state_to in self._states:
                    xi[index][state_from][state_to] /= prob_sum

        for state in self._states:
            # update start probability
            self._start_prob[state] = gamma[0][state]

            # update transition probability
            gamma_sum = 0
            for index in xrange(length - 1):
                gamma_sum += gamma[index][state]

            if gamma_sum > 0:
                xi_sum = 0
                for state_to in self._states:
                    for index in xrange(length - 1):
                        xi_sum += xi[index][state][state_to]
                    self._trans_prob[state][state_to] = xi_sum / gamma_sum
            else:
                for state_to in self._states:
                    self._trans_prob[state][state_to] = 0

            # update emission probability
            emit_gamma_sum = {}
            for index in xrange(length):
                if sequence[index] not in emit_gamma_sum:
                    emit_gamma_sum[sequence[index]] = 0
                emit_gamma_sum[sequence[index]] += gamma[index][state]

            if gamma_sum > 0:
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = emit_gamma_sum[symbol] / gamma_sum
            else:
                for symbol in self._symbols:
                    self._emit_prob[state][symbol] = 0

