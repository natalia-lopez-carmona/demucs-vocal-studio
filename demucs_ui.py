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

# ── Song analysis ──────────────────────────────────────────────────────────────

def analyze_song(file_obj):
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    path = extract_audio(path)
    if not path:
        return "—","—","—","—"
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = f"{float(np.atleast_1d(tempo)[0]):.1f} BPM"
        mins, secs = divmod(int(librosa.get_duration(y=y, sr=sr)), 60)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        mean_c = np.mean(chroma, axis=1)
        best_key, best_corr = None, -np.inf
        for i in range(12):
            for prof, mode in [(MAJOR_P,"Mayor"),(MINOR_P,"menor")]:
                corr = np.corrcoef(mean_c, np.roll(prof,i))[0,1]
                if corr > best_corr:
                    best_corr, best_key = corr, f"{NOTES[i]} {mode}"
        return bpm, f"{mins}:{secs:02d}", best_key, CAMELOT.get(best_key,"—")
    except Exception as e:
        return "Error","Error",str(e),"—"

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
}

def mix_stems(stems, selected, file_obj):
    if not selected:
        return None
    paths = [stems.get(k) for k in selected if stems.get(k) and Path(stems[k]).exists()]
    if not paths:
        return None

    datas, sr_ref = [], None
    for p in paths:
        data, sr = sf.read(p, always_2d=True)
        sr_ref = sr_ref or sr
        datas.append(data)

    max_len  = max(d.shape[0] for d in datas)
    channels = max(d.shape[1] for d in datas)
    mix = np.zeros((max_len, channels), dtype=np.float64)
    for d in datas:
        mix[:d.shape[0], :d.shape[1]] += d

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
        return *empty, "Sube una canción primero.", "—","—","—","—", {}

    raw_path = file_obj.name if hasattr(file_obj, "name") else file_obj

    progress(0, desc="Preparando…")
    audio_path = extract_audio(raw_path)
    if not audio_path:
        return *empty, "No se pudo leer el archivo.", "—","—","—","—", {}

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
            return *empty, f"Error Demucs:\n{combined[-600:]}", "—","—","—","—", {}

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
                note_min, note_max, span, voice_type,
                stems)

    except Exception as e:
        return *empty, f"Excepción: {e}", "—","—","—","—", {}

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Vocal Studio", theme=gr.themes.Soft()) as demo:

    stems_state = gr.State({})

    gr.Markdown("# Vocal Studio")

    with gr.Row():

        # ── Left panel ───────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.File(
                label="Canción (MP3, WAV, FLAC, OGG, M4A, MP4, WebM…)",
                file_types=[".mp3", ".wav", ".flac", ".ogg", ".m4a",
                            ".mp4", ".webm", ".mkv", ".mov"],
            )

            gr.Markdown("### Análisis")
            with gr.Row():
                out_bpm      = gr.Textbox(label="Tempo",    interactive=False)
                out_duration = gr.Textbox(label="Duración", interactive=False)
            with gr.Row():
                out_key     = gr.Textbox(label="Tonalidad", interactive=False)
                out_camelot = gr.Textbox(label="Camelot",   interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Separación de stems")
            model_input = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="htdemucs_6s — 6 stems (+ guitarra y piano)",
                label="Modelo",
            )
            btn_sep = gr.Button("Separar", variant="primary", size="lg")
            log_sep = gr.Textbox(label="Log", lines=3, interactive=False)

        # ── Right panel ──────────────────────────────────────────────────────────
        with gr.Column(scale=2):

            gr.Markdown("### Voz")
            out_vocals = gr.Audio(label="Voz")

            gr.Markdown("#### Rango vocal detectado")
            with gr.Row():
                out_note_min   = gr.Textbox(label="Nota más grave",           interactive=False)
                out_note_max   = gr.Textbox(label="Nota más aguda",           interactive=False)
                out_span       = gr.Textbox(label="Extensión",                interactive=False)
                out_voice_type = gr.Textbox(label="Tipo vocal de la canción", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Instrumentos")
            with gr.Row():
                out_drums  = gr.Audio(label="Batería")
                out_bass   = gr.Audio(label="Bajo")
            with gr.Row():
                out_guitar = gr.Audio(label="Guitarra")
                out_piano  = gr.Audio(label="Piano")
            out_other = gr.Audio(label="Otros")

            gr.Markdown("---")
            with gr.Row():
                btn_zip = gr.Button("Descargar todos los stems en ZIP", variant="secondary")
                zip_dl  = gr.File(label="ZIP de stems")

            gr.Markdown("---")
            gr.Markdown("### Mezcla personalizada")
            gr.Markdown("Elige qué pistas combinar (p. ej. voz + bajo) y descárgalas como un solo archivo.")
            mix_select = gr.CheckboxGroup(
                choices=[("Voz","vocals"), ("Batería","drums"), ("Bajo","bass"),
                         ("Guitarra","guitar"), ("Piano","piano"), ("Otros","other")],
                label="Pistas a combinar",
            )
            with gr.Row():
                btn_mix = gr.Button("Mezclar y descargar selección", variant="secondary")
                mix_dl  = gr.File(label="Mezcla descargable")

    # ── Events ───────────────────────────────────────────────────────────────────

    audio_input.change(
        fn=analyze_song, inputs=[audio_input],
        outputs=[out_bpm, out_duration, out_key, out_camelot])

    btn_sep.click(
        fn=separate,
        inputs=[audio_input, model_input],
        outputs=[out_vocals, out_drums, out_bass, out_other,
                 out_guitar, out_piano, log_sep,
                 out_note_min, out_note_max, out_span, out_voice_type,
                 stems_state])

    btn_zip.click(
        fn=lambda stems, f: create_stems_zip(stems, f),
        inputs=[stems_state, audio_input],
        outputs=[zip_dl])

    btn_mix.click(
        fn=mix_stems,
        inputs=[stems_state, mix_select, audio_input],
        outputs=[mix_dl])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
