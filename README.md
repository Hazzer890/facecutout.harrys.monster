# Face cutout

Drop a photo, get each head in it as a PNG with a transparent background.

Everything runs client-side. The photo never leaves your device.

## How it works

MediaPipe's face landmarker finds up to ten faces. For each one, a box around the face (with room for hair) is cropped and run through MediaPipe's multiclass selfie segmenter, which scores every pixel as hair, face skin, body skin, clothes, accessory or background. Hair, face skin and accessories are kept; body skin is kept above the lips (ears) and fades out by the chin (neck). The face outline from the landmarker is always kept, and the result is trimmed and exported as PNG.

Hair longer than the box (60% of the face height below the chin) is cut off at the box edge.

## Stack

Plain HTML, CSS and JavaScript, no build step. `css/theme.css` is the shared harrys.monster tool theme, identical across the tool subdomains. `@mediapipe/tasks-vision` 0.10.21 (ES module bundle and SIMD wasm) the float16 `face_landmarker.task` model and `selfie_multiclass_256x256.tflite` are vendored under `vendor/`, so the CSP can stay `'self'`; the wasm needs `'wasm-unsafe-eval'`.

## Developing

```bash
python3 -m http.server   # then open http://localhost:8000
```

## Deployment

Cloudflare Worker with static assets, custom domain declared in `wrangler.jsonc`. `npx wrangler deploy`, same as `metadata.harrys.monster` (see its `docs/DEPLOY.md`).
