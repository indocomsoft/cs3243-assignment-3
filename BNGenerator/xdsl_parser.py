#!/usr/bin/env python3

import json
import sys
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from typing import List
from itertools import product


class Node:
    name: str
    states: List[str]
    parents: List[str]
    probabilities: List[float]

    def __init__(self, element: Element):
        self.name = element.attrib['id']
        self.states = [
            state.attrib['id'] for state in element.findall('state')
        ]
        self.parents = Node.parse_parents(element)
        self.probabilities = Node.parse_probabilities(element)

    @staticmethod
    def parse_parents(element: Element):
        raw_parents = element.find('parents')
        if raw_parents is None:
            return []
        parents_text = raw_parents.text
        if not parents_text:
            return []
        return parents_text.split(' ')

    @staticmethod
    def parse_probabilities(element: Element):
        raw_probabilities = element.find('probabilities')
        if raw_probabilities is None:
            raise Exception("Invalid xdsl: missing probabilities")
        probabilities_text = raw_probabilities.text
        if not probabilities_text:
            raise Exception("Invalid xdsl: missing probabilities text")
        return [float(x) for x in probabilities_text.split(' ')]


class BayesianNetwork:
    def __init__(self, filename: str):
        nodes = ET.parse('graph1.xdsl').getroot().find('nodes')
        if not nodes:
            raise Exception("Invalid xdsl file")
        self.nodes = [Node(x) for x in nodes.findall('cpt')]

    def structure(self):
        variables = {node.name: node.states for node in self.nodes}
        dependencies = {
            node.name: node.parents
            for node in self.nodes if node.parents
        }
        return {"variables": variables, "dependencies": dependencies}

    def values(self):
        node_dict = {node.name: node for node in self.nodes}

        def process_root(node: Node):
            return dict(zip(node.states, node.probabilities))

        def process_non_root(node: Node):
            combinations = product(*(node_dict[p].states
                                     for p in (node.parents + [node.name])))
            states = [
                dict(zip(node.parents + ["own_value"], state))
                for state in combinations
            ]
            return [{
                **s,
                **{
                    "probability": p
                }
            } for s, p in zip(states, node.probabilities)]

        prior_probabilities = {
            node.name: process_root(node)
            for node in self.nodes if not node.parents
        }

        conditional_probabilities = {
            node.name: process_non_root(node)
            for node in self.nodes if node.parents
        }

        return {
            "prior_probabilities": prior_probabilities,
            "conditional_probabilities": conditional_probabilities
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: {} graph.xdsl".format(sys.argv[0]))
        sys.exit(1)
    bn = BayesianNetwork(sys.argv[1])
    filename = os.path.splitext(sys.argv[1])[0]
    with open(filename + '.structure.json',
              'w') as structure, open(filename + '.values.json',
                                      'w') as values:
        json.dump(bn.structure(), structure)
        json.dump(bn.values(), values)


if __name__ == "__main__":
    main()
