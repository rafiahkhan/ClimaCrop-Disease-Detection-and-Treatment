"""
Main application entry point for Crop Disease Detection API.
Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8001,  # Changed to 8001 to avoid conflict
        reload=True,
        log_level="info"
    )
