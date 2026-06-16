import gradio as gr
import subprocess
import tempfile
import zipfile
import os
import re
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# ── Constants ──────────────────────────────────────────────────────────────────

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
SOLFA = ['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si']

CAMELOT = {
    "C Mayor":"8B","C menor":"5A","C# Mayor":"3B","C# menor":"12A",
    "D Mayor":"10B","D menor":"7A","D# Mayor":"5B","D# menor":"2A",
    "E Mayor":"12B","E menor":"9A","F Mayor":"7B","F menor":"4A",
    "F# Mayor":"2B","F# menor":"11A","G Mayor":"9B","G menor":"6A",
    "G# Mayor":"4B","G# menor":"1A","A Mayor":"11B","A menor":"8A",
    "A# Mayor":"6B","A# menor":"3A","B Mayor":"1B","B menor":"10A",
}
MAJOR_P = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_P = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

VOICE_TYPES = {
    "Soprano":(60,84),"Mezzo-soprano":(57,81),"Contralto":(53,77),
    "Tenor":(48,72),"Barítono":(45,69),"Bajo":(40,64),
}

MODELS = {
    "htdemucs — 4 stems (rápido)":               "htdemucs",
    "htdemucs_ft — 4 stems (mejor calidad)":      "htdemucs_ft",
    "htdemucs_6s — 6 stems (+ guitarra y piano)": "htdemucs_6s",
    "mdx_extra_q — 4 stems (modelo clásico)":     "mdx_extra_q",
}

VIDEO_EXTS = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v'}
PCT_RE = re.compile(r'(\d+)%')

def midi_to_solfa(m):
    return f"{SOLFA[m%12]}{m//12-1}"

# ── Audio extraction (video → wav) ────────────────────────────────────────────

def extract_audio(path):
    if not path:
        return None
    if Path(path).suffix.lower() in VIDEO_EXTS:
        tmp = Path(tempfile.mkdtemp())
        out = str(tmp / "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", out],
            capture_output=True,
        )
        return out if Path(out).exists() else path
    return path

# ── HTML rendering helpers ────────────────────────────────────────────────────

def render_metrics_html(bpm="—", duration="—", key="—", camelot="—"):
    return f"""
    <div class="metrics-grid">
      <div class="metric-mini"><div class="value">{bpm}</div><div class="label">Tempo</div></div>
      <div class="metric-mini"><div class="value">{duration}</div><div class="label">Duración</div></div>
      <div class="metric-mini"><div class="value">{key}</div><div class="label">Tonalidad</div></div>
      <div class="metric-mini"><div class="value">{camelot}</div><div class="label">Camelot</div></div>
    </div>
    """

def render_range_html(note_min="—", note_max="—", span="—", voice_type="Separa una canción para ver tu rango vocal."):
    return f"""
    <div class="range-grid">
      <div class="badge badge-rose"><span class="badge-label">Nota más grave</span><span class="badge-value">{note_min}</span></div>
      <div class="badge badge-rose"><span class="badge-label">Nota más aguda</span><span class="badge-value">{note_max}</span></div>
      <div class="badge badge-lemon"><span class="badge-label">Extensión</span><span class="badge-value">{span}</span></div>
      <div class="badge badge-lemon"><span class="badge-label">Tipo vocal</span><span class="badge-value">{voice_type}</span></div>
    </div>
    """

# ── Song analysis ──────────────────────────────────────────────────────────────

def analyze_song(file_obj):
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    path = extract_audio(path)
    if not path:
        return render_metrics_html(), {}
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_val = float(np.atleast_1d(tempo)[0])
        bpm = f"{bpm_val:.1f} BPM"
        duration_sec = librosa.get_duration(y=y, sr=sr)
        mins, secs = divmod(int(duration_sec), 60)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        mean_c = np.mean(chroma, axis=1)
        best_key, best_corr = None, -np.inf
        for i in range(12):
            for prof, mode in [(MAJOR_P,"Mayor"),(MINOR_P,"menor")]:
                corr = np.corrcoef(mean_c, np.roll(prof,i))[0,1]
                if corr > best_corr:
                    best_corr, best_key = corr, f"{NOTES[i]} {mode}"
        info = {"bpm": bpm_val, "duration": duration_sec}
        camelot = CAMELOT.get(best_key, "—")
        return render_metrics_html(bpm, f"{mins}:{secs:02d}", best_key, camelot), info
    except Exception as e:
        return render_metrics_html("Error", "Error", str(e), "—"), {}

# ── Vocal range ────────────────────────────────────────────────────────────────

def analyze_vocal_range(vocals_path):
    if not vocals_path:
        return "—","—","—","Primero separa los stems."
    try:
        y, sr = librosa.load(vocals_path, sr=22050, mono=True)
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        valid = f0[voiced & ~np.isnan(f0)]
        if len(valid) == 0:
            return "—","—","—","No se detectaron notas."
        mn = int(np.round(librosa.hz_to_midi(np.percentile(valid, 5))))
        mx = int(np.round(librosa.hz_to_midi(np.percentile(valid, 95))))
        best_t, best_ov = "Desconocido", -1
        for vt,(vmin,vmax) in VOICE_TYPES.items():
            ov = max(0, min(mx,vmax)-max(mn,vmin))
            if ov > best_ov:
                best_ov, best_t = ov, vt
        return midi_to_solfa(mn), midi_to_solfa(mx), f"{mx-mn} semitonos", best_t
    except Exception as e:
        return "Error","Error","—",str(e)

# ── ZIP of all stems ───────────────────────────────────────────────────────────

def create_stems_zip(stems, file_obj):
    valid = {k: v for k, v in stems.items() if v and Path(v).exists()}
    if not valid:
        return None
    song_name = file_obj.name if hasattr(file_obj, "name") else (file_obj or "stems")
    tmp  = Path(tempfile.mkdtemp())
    name = Path(song_name).stem
    path = str(tmp / f"{name}_stems.zip")
    label_map = {
        "vocals":"voz","drums":"bateria",
        "bass":"bajo","guitar":"guitarra","piano":"piano","other":"otros",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in valid.items():
            zf.write(v, f"{label_map.get(k,k)}.wav")
    return path

# ── Custom mix of selected stems ──────────────────────────────────────────────

STEM_LABELS = {
    "vocals":"voz","drums":"bateria","bass":"bajo",
    "guitar":"guitarra","piano":"piano","other":"otros",
    "metronome":"metronomo",
}
METRONOME_VOLUME = 0.2

def generate_metronome(bpm, duration_sec, sr=44100):
    if not bpm or bpm <= 0 or duration_sec <= 0:
        return None
    interval   = 60.0 / bpm
    n_samples  = int(duration_sec * sr)
    track      = np.zeros(n_samples)
    click_len  = int(0.03 * sr)
    t          = np.linspace(0, 0.03, click_len, endpoint=False)
    click      = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 90)
    beat_time = 0.0
    while beat_time < duration_sec:
        start = int(beat_time * sr)
        end   = min(start + click_len, n_samples)
        if end > start:
            track[start:end] += click[:end - start]
        beat_time += interval
    return track

def mix_stems(stems, selected, file_obj, song_info):
    if not selected:
        return None

    use_metronome = "metronome" in selected
    real_keys = [k for k in selected if k != "metronome"]
    paths = [stems.get(k) for k in real_keys if stems.get(k) and Path(stems[k]).exists()]
    if not paths and not use_metronome:
        return None

    datas, sr_ref = [], None
    for p in paths:
        data, sr = sf.read(p, always_2d=True)
        sr_ref = sr_ref or sr
        datas.append(data)

    if datas:
        max_len  = max(d.shape[0] for d in datas)
        channels = max(d.shape[1] for d in datas)
    else:
        sr_ref   = sr_ref or 44100
        max_len  = int((song_info or {}).get("duration", 0) * sr_ref)
        channels = 1

    mix = np.zeros((max_len, channels), dtype=np.float64)
    for d in datas:
        mix[:d.shape[0], :d.shape[1]] += d

    if use_metronome and max_len > 0:
        bpm = (song_info or {}).get("bpm")
        click = generate_metronome(bpm, max_len / sr_ref, sr_ref)
        if click is not None:
            peak = np.max(np.abs(click))
            if peak > 0:
                click = click / peak * METRONOME_VOLUME
            mix[:len(click), :] += click[:max_len, None]

    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak

    song_name = file_obj.name if hasattr(file_obj, "name") else (file_obj or "mezcla")
    tmp   = Path(tempfile.mkdtemp())
    names = "_".join(STEM_LABELS.get(k, k) for k in selected)
    out_path = str(tmp / f"{Path(song_name).stem}_{names}.wav")
    sf.write(out_path, mix, sr_ref)
    return out_path

# ── Stem separation ────────────────────────────────────────────────────────────

def separate(file_obj, model_label, progress=gr.Progress()):
    empty = [None] * 6
    if not file_obj:
        return *empty, "Sube una canción primero.", render_range_html(), {}

    raw_path = file_obj.name if hasattr(file_obj, "name") else file_obj

    progress(0, desc="Preparando…")
    audio_path = extract_audio(raw_path)
    if not audio_path:
        return *empty, "No se pudo leer el archivo.", render_range_html(), {}

    model   = MODELS[model_label]
    tmp     = Path(tempfile.mkdtemp())
    utf8env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    try:
        progress(0.05, desc=f"Iniciando Demucs ({model})…")

        # Stream stdout+stderr together to parse Demucs' tqdm progress
        proc = subprocess.Popen(
            ["python", "-m", "demucs", "-n", model, "--out", str(tmp), audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=utf8env,
        )

        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            m = PCT_RE.search(line)
            if m:
                pct = min(int(m.group(1)), 100)
                # Map Demucs 0–100% → our bar 5–82%
                progress(0.05 + pct * 0.77 / 100, desc=f"Separando stems… {pct}%")

        proc.wait()
        combined = "".join(output_lines)

        if proc.returncode != 0:
            return *empty, f"Error Demucs:\n{combined[-600:]}", render_range_html(), {}

        progress(0.84, desc="Leyendo pistas…")
        stem_dir = tmp / model / Path(audio_path).stem
        s = {f.stem: str(f) for f in stem_dir.glob("*.wav")}

        vocals = s.get("vocals")
        drums  = s.get("drums")
        bass   = s.get("bass")
        other  = s.get("other")
        guitar = s.get("guitar")
        piano  = s.get("piano")

        progress(0.90, desc="Analizando rango vocal…")
        note_min, note_max, span, voice_type = analyze_vocal_range(vocals)

        stems = {"vocals":vocals, "drums":drums,
                 "bass":bass, "other":other, "guitar":guitar, "piano":piano}

        progress(1.0, desc="✓ Listo")
        return (vocals, drums, bass, other, guitar, piano,
                "✓ Stems separados.",
                render_range_html(note_min, note_max, span, voice_type),
                stems)

    except Exception as e:
        return *empty, f"Excepción: {e}", render_range_html(), {}

# ── UI ─────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Sora:wght@400;500;600;700;800&display=swap');

:root {
  --night: #1A1D21;
  --surface: #232730;
  --rose: #FF206E;
  --lemon: #FBFF12;
  --gray-light: #F5F7FA;
  --gray-mid: #B0B7C3;
  --gray-dark: #2A2F38;
}

.gradio-container { background: var(--night) !important; font-family: 'Sora', sans-serif !important; }
body { background: var(--night) !important; }

h1, h2, h3, .step-title, .card-title, .metric-mini .value, .badge-value {
  font-family: 'Space Grotesk', sans-serif !important;
}

/* Hero */
.hero {
  background: var(--surface);
  border: 1px solid rgba(255,255,255,0.08);
  border-left: 3px solid var(--rose);
  border-radius: 24px;
  padding: 32px 36px;
  margin-bottom: 24px;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #fff; margin: 10px 0 6px; letter-spacing: -0.02em; }
.hero p { color: var(--gray-mid); font-size: 0.98rem; margin: 0; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(255,32,110,0.16); color: var(--rose);
  border: 1px solid rgba(255,32,110,0.4);
  padding: 5px 14px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
}
.pulse-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--rose);
  animation: pulse 1.6s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(255,32,110,0.55); }
  70%  { box-shadow: 0 0 0 8px rgba(255,32,110,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,32,110,0); }
}

/* Cards */
.card {
  background: var(--surface) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 20px !important;
  padding: 22px !important;
  margin-bottom: 18px !important;
}
.step-title {
  font-size: 0.95rem; font-weight: 700; color: #fff;
  display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
}
.step-num {
  background: var(--lemon); color: var(--night);
  width: 24px; height: 24px; border-radius: 50%; flex: 0 0 24px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem; font-weight: 800;
}
.card-title {
  font-size: 1.05rem; font-weight: 700; color: #fff; margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}

.card-vocal {
  border: 1px solid rgba(255,32,110,0.35) !important;
  border-left: 3px solid var(--rose) !important;
}

/* Metrics & badges */
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.metric-mini {
  background: var(--gray-dark); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 14px; text-align: center;
}
.metric-mini .value { font-size: 1.5rem; font-weight: 700; color: var(--lemon); }
.metric-mini .label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--gray-mid); margin-top: 4px; }

.range-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.badge { border-radius: 14px; padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,0.08); background: var(--gray-dark); }
.badge-rose  { border-color: rgba(255,32,110,0.35); }
.badge-lemon { border-color: rgba(251,255,18,0.3); }
.badge-label { display: block; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--gray-mid); margin-bottom: 6px; }
.badge-value { display: block; font-size: 1.2rem; font-weight: 700; color: #fff; }
.badge-rose  .badge-value { color: var(--rose); }
.badge-lemon .badge-value { color: var(--lemon); }

.instrument-card { text-align: center; }
.instrument-icon { font-size: 1.5rem; margin-bottom: 4px; }

/* Primary CTA glow */
#btn-sep button {
  box-shadow: 0 0 28px rgba(251,255,18,0.35) !important;
  font-weight: 700 !important;
}

.console-log textarea {
  font-family: 'Space Grotesk', monospace !important;
  background: var(--gray-dark) !important;
  border-radius: 14px !important;
}
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Sora"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("Space Grotesk"), "ui-monospace"],
).set(
    body_background_fill="#1A1D21",
    body_background_fill_dark="#1A1D21",
    block_background_fill="#232730",
    block_background_fill_dark="#232730",
    block_border_color="rgba(255,255,255,0.07)",
    block_border_color_dark="rgba(255,255,255,0.07)",
    block_label_text_color="#B0B7C3",
    block_title_text_color="#FFFFFF",
    body_text_color="#F5F7FA",
    body_text_color_subdued="#B0B7C3",
    button_primary_background_fill="#FBFF12",
    button_primary_background_fill_hover="#e3e700",
    button_primary_text_color="#0C0F0A",
    button_secondary_background_fill="#2A2F38",
    button_secondary_background_fill_hover="#3a4150",
    button_secondary_text_color="#FFFFFF",
    input_background_fill="#2A2F38",
    input_background_fill_dark="#2A2F38",
    border_color_primary="rgba(255,255,255,0.09)",
    block_radius="20px",
    button_large_radius="14px",
    button_small_radius="10px",
    panel_background_fill="#1A1D21",
)

with gr.Blocks(title="Vocal Studio", theme=THEME, css=CUSTOM_CSS) as demo:

    stems_state     = gr.State({})
    song_info_state = gr.State({})

    gr.HTML("""
    <div class="hero">
      <div class="hero-badge"><span class="pulse-dot"></span> IA ACTIVA</div>
      <h1>Vocal Studio</h1>
      <p>Separación de stems y análisis vocal asistido por inteligencia artificial — sube una canción y descubre tu rango.</p>
    </div>
    """)

    with gr.Row():

        # ── Left panel: workflow ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="step-title"><span class="step-num">1</span>Subir canción</div>')
                audio_input = gr.File(
                    label="MP3, WAV, FLAC, OGG, M4A, MP4, WebM…",
                    file_types=[".mp3", ".wav", ".flac", ".ogg", ".m4a",
                                ".mp4", ".webm", ".mkv", ".mov"],
                )

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="step-title"><span class="step-num">2</span>Configuración</div>')
                model_input = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="htdemucs_6s — 6 stems (+ guitarra y piano)",
                    label="Modelo de separación",
                )

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="step-title"><span class="step-num">3</span>Procesar</div>')
                btn_sep = gr.Button("Separar pistas", variant="primary", size="lg", elem_id="btn-sep")
                log_sep = gr.Textbox(label="Estado", lines=3, interactive=False, elem_classes=["console-log"])

        # ── Right panel: results ─────────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">📊 Análisis musical</div>')
                metrics_html = gr.HTML(render_metrics_html())

            with gr.Group(elem_classes=["card", "card-vocal"]):
                gr.HTML('<div class="card-title">🎤 Voz</div>')
                out_vocals = gr.Audio(label="Voz aislada")
                gr.HTML('<div class="card-title" style="margin-top:18px;">Rango vocal detectado</div>')
                range_html = gr.HTML(render_range_html())

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">🎛️ Instrumentos</div>')
                with gr.Row():
                    with gr.Column(elem_classes=["instrument-card"]):
                        gr.HTML('<div class="instrument-icon">🥁</div>')
                        out_drums = gr.Audio(label="Batería")
                    with gr.Column(elem_classes=["instrument-card"]):
                        gr.HTML('<div class="instrument-icon">🎸</div>')
                        out_bass = gr.Audio(label="Bajo")
                with gr.Row():
                    with gr.Column(elem_classes=["instrument-card"]):
                        gr.HTML('<div class="instrument-icon">🎸</div>')
                        out_guitar = gr.Audio(label="Guitarra")
                    with gr.Column(elem_classes=["instrument-card"]):
                        gr.HTML('<div class="instrument-icon">🎹</div>')
                        out_piano = gr.Audio(label="Piano")
                with gr.Row():
                    with gr.Column(elem_classes=["instrument-card"]):
                        gr.HTML('<div class="instrument-icon">🎶</div>')
                        out_other = gr.Audio(label="Otros")

                with gr.Row():
                    btn_zip = gr.Button("Descargar todos los stems en ZIP", variant="secondary")
                    zip_dl  = gr.File(label="ZIP de stems")

            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-title">🎚️ Mezcla personalizada</div>')
                gr.Markdown("Elige qué pistas combinar (p. ej. voz + bajo) y descárgalas como un solo archivo.")
                mix_select = gr.CheckboxGroup(
                    choices=[("Voz","vocals"), ("Batería","drums"), ("Bajo","bass"),
                             ("Guitarra","guitar"), ("Piano","piano"), ("Otros","other"),
                             ("Metrónomo (según tempo detectado)","metronome")],
                    label="Pistas a combinar",
                )
                with gr.Row():
                    btn_mix = gr.Button("Mezclar y descargar selección", variant="secondary")
                    mix_dl  = gr.File(label="Mezcla descargable")

    # ── Events ───────────────────────────────────────────────────────────────────

    audio_input.change(
        fn=analyze_song, inputs=[audio_input],
        outputs=[metrics_html, song_info_state])

    btn_sep.click(
        fn=separate,
        inputs=[audio_input, model_input],
        outputs=[out_vocals, out_drums, out_bass, out_other,
                 out_guitar, out_piano, log_sep,
                 range_html, stems_state])

    btn_zip.click(
        fn=lambda stems, f: create_stems_zip(stems, f),
        inputs=[stems_state, audio_input],
        outputs=[zip_dl])

    btn_mix.click(
        fn=mix_stems,
        inputs=[stems_state, mix_select, audio_input, song_info_state],
        outputs=[mix_dl])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
