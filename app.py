# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
# ]
# ///

from fastapi import FastAPI, File, UploadFile
from typing import List
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

@app.post("/api/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Find questions.txt
        questions_file = next((f for f in files if f.filename == "questions.txt"), None)
        if not questions_file:
            return JSONResponse(status_code=400, content={"error": "questions.txt is required"})
        questions_text = (await questions_file.read()).decode("utf-8")
        # Save other attachments
        attachments = {}
        for f in files:
            if f.filename != "questions.txt":
                content = await f.read()
                # Save to a unique temp file per request
                temp_path = os.path.join("temp_attachments", f.filename)
                os.makedirs("temp_attachments", exist_ok=True)
                with open(temp_path, "wb") as out:
                    out.write(content)
                attachments[f.filename] = temp_path
        breakdown = task_breakdown(questions_text)
        output = generate_and_run_code(questions_text, breakdown, attachments)
        # Parse questions and build response
        import re, json
        questions = re.findall(r'^[0-9]+\. (.+)$', questions_text, re.MULTILINE)
        if not questions:
            # Try to parse as JSON object if not numbered
            try:
                questions = list(json.loads(questions_text).keys())
            except Exception:
                questions = ["Unknown question"]
        # Build response structure
        if isinstance(output, str) and output.strip().startswith("{") and output.strip().endswith("}"):
            # If output is a JSON object, return as is
            return JSONResponse(content=json.loads(output))
        else:
            # Otherwise, return as array of strings
            answers = [output for _ in questions]
            return JSONResponse(content=answers)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- Generalized code generation and execution ---
def generate_and_run_code(task: str, breakdown: str, attachments: dict = None) -> str:
    """Generate code using LLM, execute it, and return output or error. Pass attachment info to LLM."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    attach_info = f"Attachments: {list(attachments.keys()) if attachments else []}\n" if attachments else ""
    code_prompt = f"{attach_info}Given this breakdown:\n{breakdown}\nGenerate Python code to solve the task:\n{task}"
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[code_prompt],
    )
    code = response.text
    script_path = f"generated_script_{os.getpid()}.py"
    with open(script_path, "w") as f:
        f.write(code)
    import subprocess
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    retries = 2
    while result.returncode != 0 and retries > 0:
        error_prompt = f"The following code:\n{code}\nProduced this error:\n{result.stderr}\nPlease correct the code."
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[error_prompt],
        )
        code = response.text
        with open(script_path, "w") as f:
            f.write(code)
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        retries -= 1
    # Clean up temp files (optional, for thread safety)
    try:
        os.remove(script_path)
    except Exception:
        pass
    return result.stdout if result.returncode == 0 else result.stderr