-- Migration: add pitcher_role + primary_position to the players table
-- Run this once in the Supabase Dashboard → SQL Editor before running
-- backfill_positions.py or re-ingesting with the updated ingest_historical.py.

ALTER TABLE players
  ADD COLUMN IF NOT EXISTS pitcher_role     text,   -- 'SP' | 'RP' (null for hitters)
  ADD COLUMN IF NOT EXISTS primary_position text;   -- 'C' | '1B' | 'IF' | 'OF' | 'UTIL' (null for pitchers)

-- Optional: add a comment for documentation
COMMENT ON COLUMN players.pitcher_role     IS 'SP if GS/G >= 0.5, RP otherwise. Derived from Lahman Pitching.csv.';
COMMENT ON COLUMN players.primary_position IS 'Chin Music scarcity group: C | 1B | IF | OF | UTIL. Derived from Lahman Fielding.csv (position with most games played).';
