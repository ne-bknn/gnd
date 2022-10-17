import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tasks import TaskMap

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/training", response_class=HTMLResponse)
def training_index(request: Request):
    return templates.TemplateResponse(
        "index_training.html", {"request": request, "tasks": TaskMap}
    )


@app.get("/training/{task_name}", response_class=HTMLResponse)
def training_task(request: Request, task_name: str):
    if task_name not in TaskMap:
        return templates.TemplateResponse("404.html", {"request": request})

    return templates.TemplateResponse(
        "task.html", {"request": request, "task": TaskMap[task_name](None, True)}
    )


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0")
