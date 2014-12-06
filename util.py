import os, random, operator
from collections import Counter

# Borrowed from CS 221 assignment

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def chooseFromDistribution(distribution):
  "Takes a list of (prob, key) pairs and samples"
  summ = 0
  for element, score in distribution:
    summ += score
  for index, pair in enumerate(distribution):
      element, score = pair
      distribution[index] = (element, 1.0 * score / summ)

  r = random.random()
  base = 0.0
  for element, prob in distribution:
    base += prob
    if r <= base: return (element, prob)