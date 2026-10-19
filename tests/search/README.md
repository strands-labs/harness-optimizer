# tests/search

Unit tests for the candidate-search control plane. These use **synthetic fixtures only** and must
never read internal data, so that they remain runnable by anyone and by upstream CI.

Acceptance tests that replay real recorded rollouts belong with whatever integration produced them, not here.
