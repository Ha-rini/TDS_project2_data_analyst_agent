# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
# ]
# ///

from fastapi import FastAPI,File, UploadFile    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os
app = FastAPI()

from dotenv import load_dotenv
load_dotenv()


app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)
def task_breakdown(task:str):
    """Breaks down a task into smaller programmable steps using Google GenAI."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    task_breakdown_file = os.path.join('prompts', "abdul_task_breakdown.txt")
    with open(task_breakdown_file, 'r') as f:
        task_breakdown_prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[task,task_breakdown_prompt],
    )
    
    with open("abdul_breaked_task.txt", "w") as f:
        f.write(response.text)

    return response.text

@app.get("/")
async def root():
    return {"message": "Hello!"}

# create a post endpoint that processes this curl request `curl -X POST "http://127.0.0.1:8000/api/" -F "file=@question.txt"`
@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")  # assuming it's a text file
        breakdown = task_breakdown(text)
        print(breakdown)
        return {"filename": file.filename, "content": text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)