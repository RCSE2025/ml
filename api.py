from fastapi import FastAPI, File, UploadFile, Form
import aiofiles
import os

from typing import List

from detector import Model

app = FastAPI()

if not os.path.exists("tmp"):
    os.makedirs("tmp")


@app.post("/moderate")
async def moderate(files: List[UploadFile], text: str = Form(None)):
    try:
        temp_files = []

        if files:
            for file in files:
                base = os.path.basename(file.filename)
                name, ext = os.path.splitext(base)

                async with aiofiles.tempfile.NamedTemporaryFile("wb", suffix=ext, prefix=name, dir="tmp", delete=False) as temp:
                    try:
                        contents = await file.read()
                        await temp.write(contents)
                    except Exception:
                        return {"message": "There was an error uploading the file"}
                    finally:
                        await file.close()

                temp_files.append(temp.name)

        return Model(temp_files, text or "").moderate()
    except Exception as ex:
        return {"message": str(ex)}
    finally:
        for file in temp_files:
            os.remove(file)
