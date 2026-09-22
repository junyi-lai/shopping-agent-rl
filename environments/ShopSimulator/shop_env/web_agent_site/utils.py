"""Paths and small helpers shared by the ShopSimulator runtime."""

from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_eval_train.json')
