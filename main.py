"""
Kesim backend — video montaj xizmati (kesish, effektlar, ixtiyoriy subtitr).

Nima qiladi:
1. Ilova orqali video/audio fayl qabul qiladi
2. Videoni kesadi (trim), rang/shovqin/tezlik effektlarini qo'llaydi
3. Xohlasa, Whisper AI orqali ovozni matnga aylantirib, subtitr sifatida qo'shadi

Ishga tushirish:
    pip install -r requirements.txt
    uvicorn main:app --reload

Eslatma: birinchi ishga tushirishda Whisper modeli avtomatik yuklab olinadi
(internet aloqasi kerak). "small" model tezroq, "medium"/"large-v3" aniqroq,
lekin sekinroq va ko'proq xotira talab qiladi.
"""

import json
import os
import subprocess
import tempfile
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Kesim Montaj API")

# Ilova (Flutter) turli manzillardan so'rov yubora olishi uchun
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veb-versiya: iPhone/Android'da o'rnatishsiz, to'g'ridan-to'g'ri Safari/Chrome
# brauzerida ishlatish uchun. Bir xil manzilda (bir xil port) ishlagani uchun
# CORS muammosi bo'lmaydi — sahifa ham, API ham shu serverdan xizmat qiladi.
STATIC_DIR = os.path.dirname(__file__) or "."
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="web-app")

# Model faqat bir marta, server ishga tushganda yuklanadi (tezlik uchun).
# "small" — tez sinov uchun yaxshi boshlang'ich nuqta.
MODEL_SIZE = os.environ.get("SHRADI_MODEL_SIZE", "small")
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        # compute_type="int8" — CPU'da ham yengil ishlashi uchun
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


@app.get("/")
def health_check():
    return {"status": "ishlayapti", "model": MODEL_SIZE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("uz")):
    """
    Video/audio faylni qabul qilib, subtitr segmentlarini qaytaradi.
    Javob formati:
    {
        "segments": [
            {"start": 0.0, "end": 2.4, "text": "Salom, bu Shradi ilovasi."},
            ...
        ]
    }
    """
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        model = get_model()
        segments, _info = model.transcribe(
            tmp_path,
            language=language,
            vad_filter=True,  # ovoz bo'lmagan joylarni avtomatik o'tkazib yuboradi
        )

        result = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ]
        return {"segments": result}
    finally:
        os.remove(tmp_path)

def _format_srt_timestamp(seconds: float) -> str:
    """Sekundni SRT formatiga o'giradi: 00:00:01,240"""
    ms_total = max(0, round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_timestamp(seg["start"])
        end = _format_srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


# "Tungi studiya" uslubi: oq matn, qalin qora chegara, pastda markazda.
# Bu FFmpeg subtitles filtriga beriladigan uslub — Flutter'dagi shradiPrimary
# rangiga mos, keyinchalik boshqa dizayn variantlari uchun ham shu joydan
# almashtirish mumkin.
SUBTITLE_STYLE = (
    "FontName=Arial,FontSize=22,Bold=1,"
    "PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,"
    "BorderStyle=3,Outline=2,Shadow=0,Alignment=2,MarginV=60"
)

# Rang uslublari — foydalanuvchiga faqat nom ko'rsatiladi (dropdown),
# texnik parametrlar shu yerda belgilangan. Yangi uslub qo'shish uchun
# shunchaki shu lug'atga yana bitta qator qo'shish kifoya.
COLOR_PRESETS: dict[str, str | None] = {
    "original": None,
    "issiq": "eq=saturation=1.15,colorbalance=rm=0.15:bm=-0.15",
    "sovuq": "eq=saturation=1.05,colorbalance=rm=-0.15:bm=0.15",
    "kontrastli": "eq=contrast=1.3:saturation=1.2",
}

MIN_SPEED = 0.5
MAX_SPEED = 2.0

# Aspect ratio (o'lcham nisbati) uchun tayyor variantlar — CapCut'dagi kabi.
# None = original, o'zgartirilmaydi. Boshqalari markazdan kesib (crop),
# kerakli nisbatga keltiradi.
ASPECT_RATIOS: dict[str, tuple[int, int] | None] = {
    "original": None,
    "9:16": (9, 16),
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:5": (4, 5),
}

# Matn overlay joylashuvi — drawtext filtridagi y= ifodasi.
TEXT_POSITIONS: dict[str, str] = {
    "top": "h*0.08",
    "middle": "(h-text_h)/2",
    "bottom": "h*0.85",
}

# Debian asosidagi konteynerlarda (Render shu turdagi image ishlatadi)
# odatda mavjud bo'ladigan shrift — matn overlay uchun.
TEXT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _escape_drawtext(text: str) -> str:
    # drawtext filtri uchun maxsus belgilarni ekranlaymiz
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )

def _build_video_filter_chain(
    *,
    color_preset: str,
    srt_path: str | None,
    speed: float,
    aspect_ratio: str = "original",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    overlay_text: str = "",
    overlay_position: str = "bottom",
) -> str:
    parts: list[str] = []

    ratio = ASPECT_RATIOS.get(aspect_ratio)
    if ratio:
        rw, rh = ratio
        r = rw / rh
        parts.append(
            f"crop='if(gt(iw/ih,{r}),ih*{r},iw)':'if(gt(iw/ih,{r}),ih,iw/{r})'"
        )

    color_filter = COLOR_PRESETS.get(color_preset)
    if color_filter:
        parts.append(color_filter)

    if brightness != 0.0 or contrast != 1.0 or saturation != 1.0:
        parts.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}")

    if overlay_text:
        escaped = _escape_drawtext(overlay_text)
        y_expr = TEXT_POSITIONS.get(overlay_position, TEXT_POSITIONS["bottom"])
        parts.append(
            f"drawtext=fontfile={TEXT_FONT_PATH}:text='{escaped}':"
            f"fontcolor=white:fontsize=42:borderw=3:bordercolor=black@0.8:"
            f"x=(w-text_w)/2:y={y_expr}"
        )

    if srt_path:
        escaped_srt = srt_path.replace("\\", "\\\\").replace(":", "\\:")
        parts.append(f"subtitles={escaped_srt}:force_style='{SUBTITLE_STYLE}'")
    if speed != 1.0:
        parts.append(f"setpts=PTS/{speed}")
    return ",".join(parts)


def _build_audio_filter_chain(*, noise_reduction: bool, speed: float) -> str:
    parts: list[str] = []
    if noise_reduction:
        # afftdn — FFT asosida shovqinni kamaytiradi, nutqqa unchalik ta'sir qilmaydi
        parts.append("afftdn=nf=-25")
    if speed != 1.0:
        parts.append(f"atempo={speed}")
    return ",".join(parts)

@app.post("/render")
async def render(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    segments: str = Form("[]"),  # JSON string: [{"start":0.0,"end":2.4,"text":"..."}]
    noise_reduction: bool = Form(False),
    color_preset: str = Form("original"),
    speed: float = Form(1.0),
    aspect_ratio: str = Form("original"),
    brightness: float = Form(0.0),
    contrast: float = Form(1.0),
    saturation: float = Form(1.0),
    overlay_text: str = Form(""),
    overlay_position: str = Form("bottom"),
):
    """
    Video faylni qabul qilib, tanlangan barcha effektlarni (subtitr,
    shovqin tozalash, rang uslubi, moslash, o'lcham nisbati, matn, tezlik)
    qo'llab, tayyor video faylni qaytaradi.
    """
    try:
        parsed_segments = json.loads(segments)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="segments JSON noto'g'ri formatda")

    if color_preset not in COLOR_PRESETS:
        raise HTTPException(status_code=400, detail=f"Noma'lum rang uslubi: {color_preset}")
    if aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail=f"Noma'lum o'lcham nisbati: {aspect_ratio}")
    speed = max(MIN_SPEED, min(MAX_SPEED, speed))
    brightness = max(-1.0, min(1.0, brightness))
    contrast = max(0.0, min(3.0, contrast))
    saturation = max(0.0, min(3.0, saturation))

    work_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    input_path = os.path.join(tmp_dir, f"shradi_in_{work_id}{suffix}")
    output_path = os.path.join(tmp_dir, f"shradi_out_{work_id}.mp4")
    srt_path = None

    with open(input_path, "wb") as f:
        f.write(await file.read())

    if parsed_segments:
        srt_path = os.path.join(tmp_dir, f"shradi_{work_id}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(_segments_to_srt(parsed_segments))

    vf = _build_video_filter_chain(
        color_preset=color_preset,
        srt_path=srt_path,
        speed=speed,
        aspect_ratio=aspect_ratio,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        overlay_text=overlay_text,
        overlay_position=overlay_position,
    )
    af = _build_audio_filter_chain(noise_reduction=noise_reduction, speed=speed)

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if vf:
        cmd += ["-vf", vf]
    if af:
        cmd += ["-af", af]
    else:
        cmd += ["-c:a", "copy"]
    cmd.append(output_path)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    def cleanup():
        for p in (input_path, srt_path, output_path):
            if p and os.path.exists(p):
                os.remove(p)

    if proc.returncode != 0:
        cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Video render qilishda xatolik: {proc.stderr[-500:]}",
        )

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="shradi_export.mp4",
        background=background_tasks,
    )


@app.post("/mix_audio")
async def mix_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    audio: UploadFile = File(...),
    music_volume: float = Form(0.5),
    original_volume: float = Form(1.0),
):
    """
    Videoga fon musiqasi qo'shadi — video o'zining ovozi bilan birga,
    yuklangan audio fayl bilan aralashtiriladi (mix). Video davomiyligidan
    uzun bo'lsa, musiqa avtomatik qisqartiriladi.
    """
    music_volume = max(0.0, min(2.0, music_volume))
    original_volume = max(0.0, min(2.0, original_volume))

    work_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    v_suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    a_suffix = os.path.splitext(audio.filename or "music.mp3")[1] or ".mp3"
    video_path = os.path.join(tmp_dir, f"shradi_mix_v_{work_id}{v_suffix}")
    audio_path = os.path.join(tmp_dir, f"shradi_mix_a_{work_id}{a_suffix}")
    output_path = os.path.join(tmp_dir, f"shradi_mix_out_{work_id}.mp4")

    with open(video_path, "wb") as f:
        f.write(await file.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    def cleanup():
        for p in (video_path, audio_path, output_path):
            if os.path.exists(p):
                os.remove(p)

    filter_complex = (
        f"[0:a]volume={original_volume}[a0];"
        f"[1:a]volume={music_volume}[a1];"
        f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-shortest",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Musiqa qo'shishda xatolik: {proc.stderr[-500:]}",
        )

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="kesim_music.mp4",
        background=background_tasks,
    )


def _probe_duration(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


@app.post("/trim")
async def trim(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start: float = Form(0.0),
    end: float = Form(...),
):
    """
    Videoni [start, end] oralig'ida kesadi (soniyalarda). Montaj ilovasining
    eng asosiy funksiyasi — kerakli qismni ajratib olish uchun.
    """
    if end <= start:
        raise HTTPException(status_code=400, detail="Tugash vaqti boshlanish vaqtidan katta bo'lishi kerak")

    work_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    input_path = os.path.join(tmp_dir, f"shradi_trim_in_{work_id}{suffix}")
    output_path = os.path.join(tmp_dir, f"shradi_trim_out_{work_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    def cleanup():
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.remove(p)

    # -ss'ni -i'dan keyin qo'yish orqali kadrga aniq (frame-accurate) kesish
    # olamiz; qayta kodlash (re-encode) tufayli biroz sekinroq, lekin natija
    # aniq bo'ladi.
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(max(0.0, start)),
        "-to", str(end),
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Video kesishda xatolik: {proc.stderr[-500:]}",
        )

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="kesim_trim.mp4",
        background=background_tasks,
    )


@app.post("/combine")
async def combine(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    transition_duration: float = Form(0.7),
):
    """
    Bir nechta video klipni ketma-ket birlashtiradi, har biri orasiga
    silliq o'tish (crossfade transition) qo'shadi. Kamida 2 ta video kerak.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Kamida 2 ta video kerak")

    work_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    input_paths: list[str] = []
    for i, f in enumerate(files):
        suffix = os.path.splitext(f.filename or "clip.mp4")[1] or ".mp4"
        p = os.path.join(tmp_dir, f"shradi_clip_{work_id}_{i}{suffix}")
        with open(p, "wb") as out:
            out.write(await f.read())
        input_paths.append(p)

    output_path = os.path.join(tmp_dir, f"shradi_combined_{work_id}.mp4")

    def cleanup():
        for p in input_paths + [output_path]:
            if os.path.exists(p):
                os.remove(p)

    try:
        durations = [_probe_duration(p) for p in input_paths]
    except (ValueError, OSError):
        cleanup()
        raise HTTPException(status_code=400, detail="Video davomiyligini aniqlab bo'lmadi")

    # xfade/acrossfade zanjirini video sonlariga qarab dinamik quramiz.
    inputs_cmd: list[str] = []
    for p in input_paths:
        inputs_cmd += ["-i", p]

    filter_parts = []
    v_label = "0:v"
    a_label = "0:a"
    cumulative = durations[0]
    for i in range(1, len(input_paths)):
        offset = max(0.0, cumulative - transition_duration)
        next_v = f"v{i}"
        next_a = f"a{i}"
        filter_parts.append(
            f"[{v_label}][{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset:.3f}[{next_v}]"
        )
        filter_parts.append(
            f"[{a_label}][{i}:a]acrossfade=d={transition_duration}[{next_a}]"
        )
        v_label, a_label = next_v, next_a
        cumulative = offset + durations[i]

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", *inputs_cmd,
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]", "-map", f"[{a_label}]",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Klip birlashtirishda xatolik: {proc.stderr[-500:]}",
        )

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="shradi_combined.mp4",
        background=background_tasks,
    )


@app.post("/speedramp")
async def speedramp(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    speeds: str = Form(...),  # JSON: [1.0, 1.5, 0.6] — videoni shuncha teng qismga bo'lib, har biriga shu tezlikni qo'llaydi
):
    """
    Videoni teng bo'laklarga bo'lib, har bir bo'lakka alohida tezlik
    qo'llaydi (masalan: boshida sekin, o'rtada tez, oxirida sekin —
    "speed ramp"). Video va ovoz sinxron holda qayta tikiladi.
    """
    try:
        speed_list = json.loads(speeds)
        if not isinstance(speed_list, list) or not speed_list:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="speeds JSON ro'yxat bo'lishi kerak, masalan [1.0, 1.5, 0.6]")

    speed_list = [max(MIN_SPEED, min(MAX_SPEED, float(s))) for s in speed_list]

    work_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    input_path = os.path.join(tmp_dir, f"shradi_ramp_in_{work_id}{suffix}")
    output_path = os.path.join(tmp_dir, f"shradi_ramp_out_{work_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    def cleanup():
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.remove(p)

    # Agar hammasi 1.0x bo'lsa, ortiqcha ishlov berishning hojati yo'q.
    if all(abs(s - 1.0) < 1e-6 for s in speed_list):
        os.replace(input_path, output_path)
    else:
        try:
            duration = _probe_duration(input_path)
        except (ValueError, OSError):
            cleanup()
            raise HTTPException(status_code=400, detail="Video davomiyligini aniqlab bo'lmadi")

        n = len(speed_list)
        segment_len = duration / n
        filter_parts = []
        concat_inputs = []
        for i, sp in enumerate(speed_list):
            start = i * segment_len
            end = duration if i == n - 1 else (i + 1) * segment_len
            v_label = f"v{i}"
            a_label = f"a{i}"
            filter_parts.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=(PTS-STARTPTS)/{sp}[{v_label}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,atempo={sp}[{a_label}]"
            )
            concat_inputs.append(f"[{v_label}][{a_label}]")

        filter_parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]")
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            cleanup()
            raise HTTPException(
                status_code=500,
                detail=f"Speed ramp qo'llashda xatolik: {proc.stderr[-500:]}",
            )

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="shradi_speedramp.mp4",
        background=background_tasks,
    )
