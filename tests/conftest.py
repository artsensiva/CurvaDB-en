"""Пакет не установлен (даже editable) — добавляем src/ в sys.path один
раз для всех тестов, вместо дублирования sys.path.insert в каждом файле."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
