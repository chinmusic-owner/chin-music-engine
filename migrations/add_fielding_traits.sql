-- Defense traits from Lahman Fielding.csv → PA engine fielder dict (RNG, HND, ARM).
-- Run in Supabase Dashboard → SQL Editor before ingest_fielding_traits.py --push.

ALTER TABLE players
  ADD COLUMN IF NOT EXISTS rng smallint,
  ADD COLUMN IF NOT EXISTS hnd smallint,
  ADD COLUMN IF NOT EXISTS arm smallint;

COMMENT ON COLUMN players.rng IS 'Range — from Lahman PO/A/E vs innings, position-year z-scored, shrunk.';
COMMENT ON COLUMN players.hnd IS 'Hands — inverted error rate z-score, shrunk.';
COMMENT ON COLUMN players.arm IS 'Arm — assists per inning z-score, shrunk.';
