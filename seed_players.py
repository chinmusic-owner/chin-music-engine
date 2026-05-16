from database import supabase

TEST_PLAYERS = [
    {
        "player_name": "Babe Ruth",
        "season_year": 1927,
        "team": "New York Yankees",
        "position": "RF",
        "contact": 72,
        "power": 98,
        "eye": 88,
        "speed": 45,
        "stuff": 0,
        "control": 0,
        "movement": 0,
    },
    {
        "player_name": "Ted Williams",
        "season_year": 1941,
        "team": "Boston Red Sox",
        "position": "LF",
        "contact": 85,
        "power": 90,
        "eye": 99,
        "speed": 60,
        "stuff": 0,
        "control": 0,
        "movement": 0,
    },
    {
        "player_name": "Pedro Martinez",
        "season_year": 2000,
        "team": "Boston Red Sox",
        "position": "SP",
        "contact": 0,
        "power": 0,
        "eye": 0,
        "speed": 0,
        "stuff": 99,
        "control": 95,
        "movement": 92,
    },
]


def seed() -> None:
    response = supabase.table("players").insert(TEST_PLAYERS).execute()
    print(f"Inserted {len(response.data)} players:")
    for player in response.data:
        print(f"  [{player['id']}] {player['player_name']} ({player['season_year']})")


if __name__ == "__main__":
    seed()
