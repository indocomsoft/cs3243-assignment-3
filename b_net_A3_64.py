import sys
import json
import itertools
from collections import defaultdict


class BayesianNetwork(object):
    def __init__(self, structure, values, queries):
        self.variables = structure["variables"]
        self.dependencies = structure["dependencies"]
        self.conditional_probabilities = values["conditional_probabilities"]
        self.prior_probabilities = values["prior_probabilities"]
        self.queries = queries
        self.answer = []

    def construct(self):
        forward = defaultdict(list)
        for dest, sources in self.dependencies.items():
            for source in sources:
                forward[source].append(dest)
        backward = dict((k, v[:]) for k, v in self.dependencies.items())
        lst = []
        s = set(self.variables.keys()) - set(self.dependencies.keys())
        while s:
            n = s.pop()
            lst.append(n)
            for m in forward[n]:
                backward[m].remove(n)
                if not backward[m]:
                    s.add(m)
        self.toposorted = lst[::-1]

        memo = {}
        for var, values in self.conditional_probabilities.items():
            d = {}
            parents = self.dependencies[var]
            for v in values:
                d[tuple(v[p]
                        for p in (parents + ['own_value']))] = v['probability']
            memo[var] = d
        self.memo = memo

    def infer(self):
        self.answer = []  # your code to find the answer
        for query in self.queries:
            self.answer.append({
                "index":
                query["index"],
                "answer":
                self.enumeration_ask(query["tofind"], query["given"])
            })
        return self.answer

    def enumeration_ask(self, tofind, given):
        tofind_vars = list(tofind.keys())
        assignments = list(
            itertools.product(*(self.variables[var] for var in tofind_vars)))
        q = []
        for assignment in assignments:
            all_assignments = dict(zip(tofind_vars, assignment))
            all_assignments.update(given)
            q.append(self.enumerate_all(self.toposorted[:], all_assignments))

        alpha = sum(q)
        q = [x / alpha for x in q]

        correct_assignment = tuple(tofind[var] for var in tofind_vars)

        for (assignment, result) in zip(assignments, q):
            if assignment == correct_assignment:
                return result

    # assignments: Dict[string, string]
    def enumerate_all(self, vars, assignments):
        """
        This function MUTATES vars
        """
        if not vars:
            return 1.0
        y = vars.pop()
        if y in assignments:
            return self.cond_probability(y, assignments) * self.enumerate_all(
                vars[:], assignments)
        else:

            def f(val):
                new_assignments = assignments.copy()
                new_assignments[y] = val
                return self.cond_probability(
                    y, new_assignments) * self.enumerate_all(
                        vars[:], new_assignments)

            return sum(f(val) for val in self.variables[y])

    def cond_probability(self, var, assignments):
        if var in self.prior_probabilities:
            return self.prior_probabilities[var][assignments[var]]
        parents = self.dependencies[var]
        key = tuple(assignments[p] for p in (parents + [var]))
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
