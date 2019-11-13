import sys
import json
import itertools
from collections import deque


class Factor(object):
    def __init__(self, **kwargs):
        if 'copy' in kwargs:
            other = kwargs['copy']
            self.bn = other.bn
            self.free_vars = other.free_vars.copy()
            self.dict = other.dict.copy()
            return
        self.bn = kwargs['bn']
        var = kwargs['var']
        assignment = kwargs['assignment']
        variables = (self.bn.dependencies[var]
                     if var in self.bn.dependencies else []) + [var]
        free_vars = variables if not assignment else [
            v for v in variables if v not in assignment
        ]
        if not free_vars:
            self.free_vars = set()
            self.dict = {
                frozenset(): self.bn.cond_probability(var, assignment)
            }
            return
        self.free_vars = set(free_vars)
        self.dict = {}
        free_vals = [self.bn.variables[v] for v in free_vars]
        free_assignments = list(itertools.product(*(free_vals)))
        for free_assignment in free_assignments:
            free_assignment = dict(zip(free_vars, free_assignment))
            all_assignments = free_assignment.copy()
            all_assignments.update(assignment)
            self.dict[frozenset(
                free_assignment.iteritems())] = self.bn.cond_probability(
                    var, all_assignments)

    def pointwise(self, other):
        common_vars = list(self.free_vars & other.free_vars)
        self.free_vars = self.free_vars | other.free_vars
        new_dict = {}
        if not common_vars:
            for k1, v1 in self.dict.iteritems():
                for k2, v2 in other.dict.iteritems():
                    new_dict[k1 | k2] = v1 * v2
        else:
            common_vals = (self.bn.variables[v] for v in common_vars)
            assignments = itertools.product(*common_vals)
            for assignment in assignments:
                pairs = set(zip(common_vars, assignment))
                for k1, v1 in ((k, v) for k, v in self.dict.iteritems()
                               if pairs <= k):
                    for k2, v2 in ((k, v) for k, v in other.dict.iteritems()
                                   if pairs <= k):
                        new_dict[k1 | k2] = v1 * v2
        self.dict = new_dict
        return self

    def eliminated(self, assignment_pair):
        other = self.copy()
        return other.eliminate(assignment_pair)

    def eliminate(self, assignment_pair):
        pair_set = frozenset([assignment_pair])
        filtered = [(k, v) for k, v in self.dict.iteritems()
                    if assignment_pair in k]
        self.dict = dict((k - pair_set, v) for k, v in filtered)
        self.free_vars.remove(assignment_pair[0])
        return self

    def add(self, other):
        for k in self.dict.iterkeys():
            self.dict[k] += other.dict[k]
        return self

    @staticmethod
    def sum_out(var, factors):
        # var must be in free_vars of all factors
        assignment_pairs = itertools.product(
            [var],
            iter(factors).next().bn.variables[var])
        eliminated = (reduce(lambda x, y: x.pointwise(y), (f.eliminated(pair)
                                                           for f in factors))
                      for pair in assignment_pairs)
        summed = reduce(lambda x, y: x.add(y), eliminated)
        return summed

    def copy(self):
        return Factor(copy=self)


class BayesianNetwork(object):
    def __init__(self, structure, values, queries):
        self.variables = structure["variables"]
        self.dependencies = structure["dependencies"]
        self.conditional_probabilities = values["conditional_probabilities"]
        self.prior_probabilities = values["prior_probabilities"]
        self.queries = queries
        self.answer = []
        self.children = {}
        self.memo = {}

    def construct(self):
        for child, parents in self.dependencies.iteritems():
            for parent in parents:
                if parent not in self.children:
                    self.children[parent] = set()
                self.children[parent].add(child)

        for var, values in self.conditional_probabilities.iteritems():
            d = {}
            parents = self.dependencies[var]
            for v in values:
                k = tuple(v[p] for p in parents + ['own_value'])
                d[k] = v['probability']
            self.memo[var] = d

    def infer(self):
        for query in self.queries:
            self.answer.append({
                "index":
                query["index"],
                "answer":
                self.elimination_ask(query["tofind"], query["given"])
            })
        return self.answer

    def elimination_ask(self, tofind, given):
        # Get relevant variables
        tofind_vars = set(tofind.iterkeys())
        relevant = set()
        for var in itertools.chain(tofind.iterkeys(), given.iterkeys()):
            stack = deque([var])
            while stack:
                v = stack.pop()
                if v in relevant:
                    continue
                relevant.add(v)
                if v in self.dependencies:
                    stack.extend(self.dependencies[v])
        free_vars = set(var for var in relevant if var not in given)
        factors = set()
        added = set()
        for free_var in free_vars:
            to_add = [free_var]
            if free_var in self.children:
                to_add = itertools.chain(to_add, self.children[free_var])
            for var in to_add:
                if var in relevant and var not in added:
                    added.add(var)
                    factors.add(Factor(bn=self, var=var, assignment=given))
        free_vars -= tofind_vars
        ordered = sorted(free_vars,
                         key=lambda v: sum(1 for factor in factors
                                           if v in factor.free_vars))
        for free_var in ordered:
            subset = set(factor for factor in factors
                         if free_var in factor.free_vars)
            factors -= subset
            summed = Factor.sum_out(free_var, subset)
            factors.add(summed)
        final_factor = reduce(lambda x, y: x.pointwise(y), factors)
        alpha = sum(final_factor.dict.itervalues())
        desired = final_factor.dict[frozenset(tofind.iteritems())]
        return desired / alpha

    def cond_probability(self, var, assignments):
        if var in self.prior_probabilities:
            return self.prior_probabilities[var][assignments[var]]
        parents = self.dependencies[var]
        key = tuple(assignments[p] for p in parents + [var])
        return self.memo[var][key]

    # You may add more classes/functions if you think is useful. However,
    # ensure all the classes/functions are in this file ONLY and used within
    # the BayesianNetwork class.


def main():
    # STRICTLY do NOT modify the code in the main function here
    if len(sys.argv) != 4:
        print("""
Usage: python b_net_A3_xx.py structure.json values.json queries.json
            """)
        raise ValueError("Wrong number of arguments!")

    structure_filename = sys.argv[1]
    values_filename = sys.argv[2]
    queries_filename = sys.argv[3]

    try:
        with open(structure_filename, 'r') as f:
            structure = json.load(f)
        with open(values_filename, 'r') as f:
            values = json.load(f)
        with open(queries_filename, 'r') as f:
            queries = json.load(f)

    except IOError:
        raise IOError("Input file not found or not a json file")

    # testing if the code works
    b_network = BayesianNetwork(structure, values, queries)
    b_network.construct()
    answers = b_network.infer()

    print(answers)


if __name__ == "__main__":
    main()
