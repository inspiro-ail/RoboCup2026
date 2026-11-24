"""Lorax AI Voice Studio: direct voice conversion with RVC + RMVPE."""

import asyncio
import logging
import re
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rvc_service import BASE_DIR, CHARACTERS, OUTPUTS_DIR, RVCNotReadyError, rvc_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoraxServer")
app = FastAPI(title="Lorax AI Voice Studio", description="RVC + RMVPE voice conversion", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
MAX_UPLOAD_SIZE = 15 * 1024 * 1024


@app.get("/images/{filename}")
def serve_image(filename: str):
    if not re.fullmatch(r"[\w-]+\.(jpg|jpeg|png)", filename, re.IGNORECASE):
        raise HTTPException(404, "Image not found")
    for folder in (STATIC_DIR / "images", BASE_DIR / "images", STATIC_DIR):
        target = folder / filename
        if target.is_file():
            return FileResponse(target)
    raise HTTPException(404, "Image not found")


@app.get("/api/status")
@app.get("/health")
def get_status():
    return rvc_service.get_system_status()


@app.post("/api/convert")
async def convert_voice(
    audio: UploadFile = File(...),
    character: str = Form("lorax"),
    pitch_shift: int = Form(0, ge=-24, le=24),
    index_rate: float = Form(0.75, ge=0.0, le=1.0),
    protect: float = Form(0.33, ge=0.0, le=0.5),
    rms_mix_rate: float = Form(0.0, ge=0.0, le=1.0),
    custom_text: Optional[str] = Form(None),
    translate_to_en: bool = Form(False),
    speed: Optional[float] = Form(None),
):
    character = character.lower().strip()
    if character not in CHARACTERS:
        raise HTTPException(400, "Выберите lorax или pipsqueak.")
    if custom_text or translate_to_en or (speed is not None and speed != 1.0):
        raise HTTPException(400, "RVC преобразует исходную речь. Перевод, замена текста и изменение темпа не поддерживаются.")
    raw = await audio.read(MAX_UPLOAD_SIZE + 1)
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "Максимальный размер записи — 15 МБ.")
    if not raw:
        raise HTTPException(400, "Пустая аудиозапись.")
    job_id = uuid.uuid4().hex
    source = TEMP_DIR / f"upload_{job_id}.audio"
    output = OUTPUTS_DIR / f"{job_id}.wav"
    source.write_bytes(raw)
    try:
        details = await asyncio.to_thread(
            rvc_service.convert_voice, character, str(source), str(output),
            pitch_shift=pitch_shift, index_rate=index_rate,
            protect=protect, rms_mix_rate=rms_mix_rate,
        )
        return {"success": True, "character": character,
                "character_name": CHARACTERS[character]["name"],
                "audio_url": f"/outputs/{job_id}.wav", **details}
    except RVCNotReadyError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("RVC conversion failed")
        raise HTTPException(500, str(exc)[:400]) from exc
    finally:
        source.unlink(missing_ok=True)


@app.get("/outputs/{filename}")
def get_output_audio(filename: str):
    if not re.fullmatch(r"[a-f0-9]{32}\.wav", filename):
        raise HTTPException(404, "Аудиофайл не найден.")
    target = OUTPUTS_DIR / filename
    if not target.is_file():
        raise HTTPException(404, "Аудиофайл не найден.")
    return FileResponse(target, media_type="audio/wav", filename="rvc_voice.wav")


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def serve_css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    print("Lorax AI Voice Studio — RVC + RMVPE: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
