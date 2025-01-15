from fastapi import FastAPI
from app.routes import scheduling
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get allowed origins from .env file
allowed_origins = os.getenv("MAIN_SERVICE", "http://localhost:3000").split(",")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dynamically set origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Register routers
app.include_router(scheduling.router, prefix="/api", tags=["Scheduling"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Scheduling API"}
