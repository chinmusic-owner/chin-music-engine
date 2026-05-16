from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def test_connection() -> None:
    """Confirm the Supabase connection is live by fetching one row from 'players'."""
    response = supabase.table("players").select("id").limit(1).execute()
    print("Connection successful!")
    print(f"Response: {response.data}")


if __name__ == "__main__":
    test_connection()
