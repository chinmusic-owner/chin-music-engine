from fastapi import FastAPI
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Connection successful!")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello Chin Music"}
