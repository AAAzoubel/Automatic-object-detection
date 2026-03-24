from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import shutil
import os
import time

from object_detector import load_model, process_video


# --------------------------------------------------
# Modelos Pydantic da resposta da API
# --------------------------------------------------

class Summary(BaseModel):
    total_detections: int
    classes_detected: List[str]
    detections_by_class: Dict[str, int]
    unique_objects_by_class: Dict[str, int]


class FilesGenerated(BaseModel):
    annotated_video: str
    csv: str
    excel: str
    stats_total: Optional[str] = None
    stats_unique: Optional[str] = None


class DetectionResponse(BaseModel):
    message: str
    filename: str
    processing_time_seconds: float
    summary: Summary
    files_generated: FilesGenerated

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Video processed successfully",
                "filename": "jogo.mp4",
                "processing_time_seconds": 12.47,
                "summary": {
                    "total_detections": 320,
                    "classes_detected": ["person", "sports ball"],
                    "detections_by_class": {
                        "person": 300,
                        "sports ball": 20
                    },
                    "unique_objects_by_class": {
                        "person": 22,
                        "sports ball": 1
                    }
                },
                "files_generated": {
                    "annotated_video": "outputs/video_detectado.mp4",
                    "csv": "outputs/deteccoes_video.csv",
                    "excel": "outputs/deteccoes_video.xlsx",
                    "stats_total": "outputs/estatisticas_deteccoes.csv",
                    "stats_unique": "outputs/estatisticas_objetos_unicos.csv"
                }
            }
        }


app = FastAPI(
    title="Video Object Detection API",
    description="""
API para upload e análise de vídeos com detecção e tracking de objetos usando YOLOv8.

### Fluxo
1. O usuário envia um vídeo
2. A API salva o arquivo na pasta `uploads/`
3. O vídeo é processado frame a frame
4. A API gera:
   - vídeo anotado
   - CSV com detecções
   - Excel com detecções
   - estatísticas por classe
5. A resposta retorna um resumo em JSON
""",
    version="1.0.0"
)

# Carrega o modelo uma única vez quando a API sobe
model = load_model()


@app.get(
    "/",
    tags=["Status"],
    summary="Verificar se a API está ativa",
    description="Endpoint simples para confirmar que a API está rodando."
)
def home():
    return {
        "message": "Video Detection API running",
        "docs": "/docs",
        "ui": "/ui"
    }


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Video Object Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 40px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: #111827;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                margin-top: 0;
                font-size: 28px;
            }
            p {
                color: #cbd5e1;
            }
            input[type="file"] {
                margin: 20px 0;
                display: block;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 15px;
            }
            button:hover {
                background: #1d4ed8;
            }
            .status {
                margin-top: 20px;
                font-weight: bold;
            }
            pre {
                background: #020617;
                padding: 16px;
                border-radius: 12px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-top: 20px;
            }
            .links {
                margin-top: 20px;
            }
            a {
                color: #60a5fa;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Object Detection</h1>
            <p>Upload a video and run object detection + tracking with YOLOv8.</p>

            <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv" />
            <button onclick="sendVideo()">Process Video</button>

            <div class="status" id="status"></div>
            <pre id="result"></pre>

            <div class="links">
                <p><a href="/docs" target="_blank">Open page Docs</a></p>
            </div>
        </div>

        <script>
            async function sendVideo() {
                const fileInput = document.getElementById("videoFile");
                const status = document.getElementById("status");
                const resultBox = document.getElementById("result");

                if (!fileInput.files.length) {
                    status.textContent = "Please select a video file first.";
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                status.textContent = "Uploading video and processing frames... this may take a while.";
                resultBox.textContent = "";

                try {
                    const response = await fetch("/detect-video", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        status.textContent = "Request failed.";
                        resultBox.textContent = JSON.stringify(data, null, 2);
                        return;
                    }

                    status.textContent = `Processing finished successfully in ${data.processing_time_seconds} seconds.`;
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    status.textContent = "Unexpected error while sending the video.";
                    resultBox.textContent = error.toString();
                }
            }
        </script>
    </body>
    </html>
    """


@app.post(
    "/detect-video",
    tags=["Detection"],
    summary="Enviar vídeo para detecção de objetos",
    description="""
Recebe um arquivo de vídeo, executa detecção + tracking com YOLOv8
e retorna um resumo das detecções, além dos caminhos dos arquivos gerados.
""",
    response_model=DetectionResponse
)
async def detect_video(
    file: UploadFile = File(
        ...,
        description="Arquivo de vídeo nos formatos: .mp4, .avi, .mov ou .mkv"
    )
):
    os.makedirs("uploads", exist_ok=True)

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nenhum arquivo foi enviado."
        )

    allowed_video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    if not file.filename.lower().endswith(allowed_video_extensions):
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie um arquivo .mp4, .avi, .mov ou .mkv."
        )

    uploaded_video_path = os.path.join("uploads", file.filename)

    with open(uploaded_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.time()

    processing_result = process_video(uploaded_video_path, model)

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    return {
        "message": "Video processed successfully",
        "filename": file.filename,
        "processing_time_seconds": processing_time,
        "summary": processing_result["summary"],
        "files_generated": {
            "annotated_video": processing_result["output_video_path"],
            "csv": processing_result["csv_path"],
            "excel": processing_result["excel_path"],
            "stats_total": processing_result["stats_total_path"],
            "stats_unique": processing_result["stats_unique_path"]
        }
    }