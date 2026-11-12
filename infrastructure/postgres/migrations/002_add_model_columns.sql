-- Migration 002: Track champion and candidate model identifiers for multi-model canary runs

ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS champion_model VARCHAR(128);
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS candidate_model VARCHAR(128);

CREATE INDEX IF NOT EXISTS idx_evaluations_models ON evaluations(champion_model, candidate_model);
