import uvicorn

if __name__ == "__main__":
    # Runs the FastAPI app from the backend directory
    # host="0.0.0.0" allows access from the local network
    # reload=True automatically restarts the server when code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="backend")
