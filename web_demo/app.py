from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from web_demo.history import HistoryStore
from web_demo.registry import WatermarkRegistry
from web_demo.service import WatermarkService


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "AudioMarkNet-demo")

    service = WatermarkService()
    history = HistoryStore(service.storage_dir / "history.sqlite3", service.storage_dir)
    registry = WatermarkRegistry(service.storage_dir / "registry.sqlite3")

    def _store_upload(file_storage) -> tuple[str, Path, str]:
        original_name = secure_filename(file_storage.filename or "audio.wav") or "audio.wav"
        suffix = Path(original_name).suffix or ".wav"
        stored_name = f"{Path(original_name).stem}_{uuid.uuid4().hex[:8]}{suffix}"
        stored_path = service.upload_dir / stored_name
        file_storage.save(stored_path)
        return original_name, stored_path, f"uploads/{stored_name}"

    def _store_output(audio, stem: str) -> tuple[Path, str]:
        out_name = f"{stem}_{uuid.uuid4().hex[:8]}.wav"
        out_path = service.output_dir / out_name
        service._save_wav(audio, out_path)
        return out_path, f"outputs/{out_name}"

    def _base_context(**kwargs):
        context = {
            "wm_length": service.wm_length,
            "records": history.list(50),
            "watermarks": registry.list(50),
            "result": None,
            "active_page": kwargs.pop("active_page", "embed"),
        }
        context.update(kwargs)
        return context

    @app.get("/")
    def index():
        return redirect(url_for("embed_page"))

    @app.get("/embed")
    def embed_page():
        return render_template("embed.html", **_base_context(active_page="embed"))

    @app.post("/embed")
    def embed():
        audio_file = request.files.get("audio_file")
        name = request.form.get("watermark_name", "")
        code = request.form.get("watermark_code", "")
        if not audio_file or not audio_file.filename:
            flash("请先上传原始音频")
            return redirect(url_for("embed_page"))

        try:
            original_name, stored_path, stored_rel = _store_upload(audio_file)
            waveform = service._load_audio(stored_path)
            package = service.train_and_save_model_package(name, code, waveform)
            reg = registry.register(name, code, model_rel_path=package["model_rel_path"])
            result = service.embed(stored_path, reg.code, model_rel_path=reg.model_rel_path)
            out_path, out_rel = _store_output(result["audio"], Path(original_name).stem)
            record_id = history.add(
                operation="embed",
                input_name=original_name,
                output_name=out_path.name,
                watermark_code=reg.code,
                detected_code=reg.name,
                confidence=result["bit_accuracy"],
                status="success",
                artifact_path=out_rel,
                metadata={
                    "input_path": stored_rel,
                    "watermark_name": reg.name,
                    "model_rel_path": reg.model_rel_path,
                    "model_metadata_rel_path": package["metadata_rel_path"],
                    "input_duration": result["input_duration"],
                    "segment_count": result["segment_count"],
                    "bit_accuracy": result["bit_accuracy"],
                    "clean_accuracy": result["clean_accuracy"],
                    "progress": package.get("progress", []),
                },
            )
            result_card = {
                "mode": "embed",
                "status": "success",
                "record_id": record_id,
                "input_name": original_name,
                "watermark_name": reg.name,
                "output_name": out_path.name,
                "download_url": url_for("download_file", rel_path=out_rel),
                "watermark_code": reg.code,
                "model_rel_path": reg.model_rel_path,
                "bit_accuracy": result["bit_accuracy"],
                "clean_accuracy": result["clean_accuracy"],
                "input_duration": result["input_duration"],
                "segment_count": result["segment_count"],
                "progress": package.get("progress", []),
            }
            flash("水印嵌入完成，并已保存对应模型")
        except Exception as exc:
            flash(f"水印嵌入失败: {exc}")
            result_card = {"mode": "embed", "status": "error", "message": str(exc)}

        return render_template("embed.html", **_base_context(active_page="embed", result=result_card))

    @app.get("/detect")
    def detect_page():
        return render_template("detect.html", **_base_context(active_page="detect"))

    @app.post("/detect")
    def detect():
        audio_file = request.files.get("audio_file")
        if not audio_file or not audio_file.filename:
            flash("请先上传待检测音频")
            return redirect(url_for("detect_page"))

        try:
            original_name, stored_path, stored_rel = _store_upload(audio_file)
            result = service.detect_against_saved_models(stored_path, registry.list(500))
            record_id = history.add(
                operation="detect",
                input_name=original_name,
                output_name=None,
                watermark_code=result.get("detected_code"),
                detected_code=result.get("detected_name"),
                confidence=result["confidence"],
                status="success",
                artifact_path=None,
                metadata={
                    "input_path": stored_rel,
                    "input_duration": result["input_duration"],
                    "segment_count": result["segment_count"],
                    "bit_agreement": result["bit_agreement"],
                    "matched": result["matched"],
                    "evaluations": result["evaluations"],
                    "raw_detected_code": result.get("raw_detected_code"),
                },
            )
            result_card = {
                "mode": "detect",
                "status": "success",
                "record_id": record_id,
                "input_name": original_name,
                **result,
            }
            flash("水印检测完成")
        except Exception as exc:
            flash(f"水印检测失败: {exc}")
            result_card = {"mode": "detect", "status": "error", "message": str(exc)}

        return render_template("detect.html", **_base_context(active_page="detect", result=result_card))

    @app.get("/history")
    def history_page():
        return render_template("history.html", **_base_context(active_page="history"))

    @app.post("/history/clear")
    def clear_history():
        history.clear()
        flash("历史记录已清空")
        return redirect(url_for("history_page"))

    @app.post("/watermarks/clear")
    def clear_watermarks():
        registry.clear()
        service.clear_model_packages()
        flash("水印注册及模型包已清空")
        return redirect(url_for("history_page"))

    @app.get("/files/<path:rel_path>")
    def download_file(rel_path: str):
        file_path = (service.storage_dir / rel_path).resolve()
        if service.storage_dir.resolve() not in file_path.parents and file_path != service.storage_dir.resolve():
            abort(404)
        if not file_path.exists():
            abort(404)
        return send_from_directory(service.storage_dir, rel_path, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
