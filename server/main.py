from fastapi import FastAPI
from auth.routes import router as auth_router
from docs.routes import router as docs_router
from chat.routes import  router as chat_router

from fastapi.staticfiles import StaticFiles
from pathlib import Path

image_dir = Path("./uploaded_docs/page_images")
image_dir.mkdir(parents=True, exist_ok=True)




app=FastAPI()

app.mount("/page_images", StaticFiles(directory=image_dir), name="page_images")

app.include_router(auth_router)
app.include_router(docs_router)
app.include_router(chat_router)

@app.get("/health")
def health_check():
    return {"message":"OK"}


# def main():
#     print("Hello from server!")


# if __name__ == "__main__":
#     main()
