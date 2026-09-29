# Subtitle Remover V1

Standalone CPU-only hard-subtitle remover inspired by the simple Wire Remover workflow.

## Features

- Separate tool; no dependency on `video-audio`
- FastAPI Web UI
- Upload video
- Drag two points across the subtitle line
- `FAST` clone mode
- `SMOOTH` vertical interpolation + feather
- Preview before processing
- Server-side background processing
- Progress polling
- Original audio re-mux
- Automatic job cleanup after 6 hours
- No Torch, CUDA, OCR, or AI model

## VPS install

```bash
mkdir -p /root/subtitle-remover
cd /root/subtitle-remover
# Copy/extract these project files here
chmod +x install.sh
./install.sh
```

Open:

```text
http://YOUR_SERVER_IP:9091
```

Health check:

```bash
curl http://127.0.0.1:9091/health
```

Logs:

```bash
journalctl -u subtitle-remover -f
```

## Notes

V1 works best when the subtitle occupies a stable horizontal band and the background directly above/below it is similar. Complex moving backgrounds should later use an AI inpainting fallback.
