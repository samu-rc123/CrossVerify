-- CrossVerify Database Initialization Script
-- Initializes extensions and tables required for telemetry and evaluation metrics

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: evaluations
-- Stores side-by-side inference outputs and telemetry metrics for Champion vs. Candidate models
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(128) NOT NULL,
    champion_model VARCHAR(128),
    candidate_model VARCHAR(128),
    prompt TEXT NOT NULL,
    champion_response TEXT,
    candidate_response TEXT,
    champion_latency_ms DOUBLE PRECISION,
    candidate_latency_ms DOUBLE PRECISION,
    semantic_similarity DOUBLE PRECISION,
    structural_valid BOOLEAN,
    discrepancy_detected BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for time-series aggregation and sliding window telemetry
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_request_id ON evaluations(request_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_discrepancy ON evaluations(discrepancy_detected);
CREATE INDEX IF NOT EXISTS idx_evaluations_models ON evaluations(champion_model, candidate_model);
