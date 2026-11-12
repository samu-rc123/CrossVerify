-- Migration 001: Initial evaluations telemetry table and indices

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(128) NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_request_id ON evaluations(request_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_discrepancy ON evaluations(discrepancy_detected);
