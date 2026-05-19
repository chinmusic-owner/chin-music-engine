-- Migration: add roster loader fields required by PRD 05
-- Run once in Supabase Dashboard → SQL Editor before re-running ingest_historical.py.
--
-- New columns:
--   pa          — plate appearances (hitters); used for lineup volume sort
--   ip          — innings pitched as decimal (pitchers); used for staff sort
--   bats        — batting handedness: 'R' | 'L' | 'B' (from Lahman People.csv)
--   throws      — throwing handedness: 'R' | 'L' (from Lahman People.csv)
--   ak          — anti-K trait (20–99); global z-score of k_plus_inv
--   gap         — gap power trait (20–99); global z-score of xbh_plus
--   data_source — provenance tag: 'historical' for Lahman-ingested rows;
--                 NULL / 'arcade' for manually-seeded test rows.
--                 load_team() filters for data_source = 'historical'.

ALTER TABLE players
  ADD COLUMN IF NOT EXISTS pa          integer,
  ADD COLUMN IF NOT EXISTS ip          real,
  ADD COLUMN IF NOT EXISTS bats        text,
  ADD COLUMN IF NOT EXISTS throws      text,
  ADD COLUMN IF NOT EXISTS ak          smallint,
  ADD COLUMN IF NOT EXISTS gap         smallint,
  ADD COLUMN IF NOT EXISTS data_source text;

COMMENT ON COLUMN players.pa          IS 'Plate appearances (hitters). AB + BB + HBP + SF + SH. Used for starting lineup sort.';
COMMENT ON COLUMN players.ip          IS 'Innings pitched as decimal (pitchers). IPouts / 3.0. Used for starter selection.';
COMMENT ON COLUMN players.bats        IS 'Batting handedness from Lahman People.csv: R | L | B.';
COMMENT ON COLUMN players.throws      IS 'Throwing handedness from Lahman People.csv: R | L.';
COMMENT ON COLUMN players.ak          IS 'Anti-K / bat-to-ball trait (20–99). Global z-score of k_plus_inv = lg_K% / player_K%.';
COMMENT ON COLUMN players.gap         IS 'Gap power trait (20–99). Global z-score of xbh_plus = player_XBH% / lg_XBH%.';
COMMENT ON COLUMN players.data_source IS 'Row provenance: historical = Lahman-ingested; arcade = manually seeded test row.';

-- Mark all pre-existing rows as arcade so they are excluded from load_team()
-- until they are overwritten by a proper re-ingestion run.
-- Rows with player_id IS NULL are definitively arcade (no Lahman ID).
UPDATE players SET data_source = 'arcade' WHERE data_source IS NULL;
