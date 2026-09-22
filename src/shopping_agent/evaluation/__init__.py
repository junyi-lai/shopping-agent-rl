"""Deterministic evaluation of Shopping Agent trajectories on Final-200 Clean.

The evaluation entry point rolls a model out on the frozen benchmark, scores every
trajectory with Reward v4 and aggregates fixed-denominator metrics; nothing here
judges a trajectory with another language model.
"""
