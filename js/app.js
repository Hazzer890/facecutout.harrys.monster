import { FaceLandmarker, FilesetResolver, ImageSegmenter } from '../vendor/vision_bundle.mjs';

// Face mesh landmarks tracing the jaw and hairline, in order around the face.
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const LIP = 13;
// selfie_multiclass_256x256 classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 accessories.
const HAIR = 1, BODY = 2, FACE = 3, ACCESSORY = 5;

const cards = document.getElementById('cards');
const tpl = document.getElementById('card-tpl');

let models;
function getModels() {
  models ??= FilesetResolver.forVisionTasks('vendor/wasm').then(fs => Promise.all([
    FaceLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: 'vendor/face_landmarker.task' },
      runningMode: 'IMAGE',
      numFaces: 10,
    }),
    ImageSegmenter.createFromOptions(fs, {
      baseOptions: { modelAssetPath: 'vendor/selfie_multiclass_256x256.tflite' },
      runningMode: 'IMAGE',
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    }),
  ]));
  return models;
}

// Cuts one head out of the image: hair, face, ears and accessories, but not neck or clothes.
// The segmenter runs on a crop around the face so its 256px input is spent on the head alone.
function cutout(img, lm, segmenter) {
  const xs = OVAL.map(n => lm[n].x * img.width), ys = OVAL.map(n => lm[n].y * img.height);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = maxX - minX, h = maxY - minY;
  // ponytail: fixed margins around the face; very big hair or hats get clipped at the box edge.
  const x0 = Math.floor(Math.max(0, minX - 0.6 * w)), x1 = Math.ceil(Math.min(img.width, maxX + 0.6 * w));
  const y0 = Math.floor(Math.max(0, minY - 0.8 * h)), y1 = Math.ceil(Math.min(img.height, maxY + 0.6 * h));
  const W = x1 - x0, H = y1 - y0;
  const crop = Object.assign(document.createElement('canvas'), { width: W, height: H });
  crop.getContext('2d').drawImage(img, -x0, -y0);

  const result = segmenter.segment(crop);
  const conf = result.confidenceMasks.map(m => m.getAsFloat32Array());
  const { width: mw, height: mh } = result.confidenceMasks[0];
  result.close();

  // Build the alpha mask at the segmenter's resolution. Body skin counts above the lips
  // (ears) and fades out by the chin (neck).
  const sx = mw / W, sy = mh / H;
  const lipY = (lm[LIP].y * img.height - y0) * sy, chinY = (maxY - y0) * sy;
  const mask = new ImageData(mw, mh);
  mask.data.fill(255);
  for (let i = 0; i < mw * mh; i++) {
    const body = Math.min(1, Math.max(0, (chinY - Math.floor(i / mw)) / (chinY - lipY)));
    const a = conf[HAIR][i] + conf[FACE][i] + conf[ACCESSORY][i] + body * conf[BODY][i];
    // Stretch 0.3–0.7 to 0–1: drops the faint background haze, keeps soft hair edges.
    mask.data[i * 4 + 3] = Math.min(1, Math.max(0, (a - 0.3) / 0.4)) * 255;
  }
  const maskCanvas = Object.assign(document.createElement('canvas'), { width: mw, height: mh });
  const mctx = maskCanvas.getContext('2d');
  mctx.putImageData(mask, 0, 0);
  // The face oval is always head, whatever the segmenter thought.
  mctx.beginPath();
  xs.forEach((x, i) => mctx.lineTo((x - x0) * sx, (ys[i] - y0) * sy));
  mctx.fillStyle = '#fff';
  mctx.fill();

  // Trim to the visible pixels, then draw the photo through the upscaled (smoothed) mask.
  const alpha = mctx.getImageData(0, 0, mw, mh).data;
  let bx0 = mw, by0 = mh, bx1 = 0, by1 = 0;
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    if (alpha[(y * mw + x) * 4 + 3] < 13) continue;
    bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x + 1); by0 = Math.min(by0, y); by1 = Math.max(by1, y + 1);
  }
  const ox = Math.floor(bx0 / sx), oy = Math.floor(by0 / sy);
  const out = Object.assign(document.createElement('canvas'), {
    width: Math.ceil(bx1 / sx) - ox, height: Math.ceil(by1 / sy) - oy,
  });
  const ctx = out.getContext('2d');
  ctx.drawImage(crop, -ox, -oy);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(maskCanvas, -ox, -oy, W, H);
  return out;
}

async function process(file) {
  const card = tpl.content.firstElementChild.cloneNode(true);
  const $ = sel => card.querySelector(sel);
  $('.card-name').textContent = file.name;
  $('.card-remove').addEventListener('click', () => card.remove());
  cards.prepend(card);

  try {
    // createImageBitmap applies EXIF orientation, so phone photos come out upright.
    const img = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const [landmarker, segmenter] = await getModels();
    const { faceLandmarks } = landmarker.detect(img);
    $('.card-summary').textContent = faceLandmarks.length === 0 ? 'No head found.'
      : `${faceLandmarks.length} head${faceLandmarks.length === 1 ? '' : 's'} found.`;
    window.track?.('cutout', { faces: faceLandmarks.length });

    const base = file.name.replace(/\.[^.]+$/, '');
    faceLandmarks.forEach((lm, i) => {
      const canvas = cutout(img, lm, segmenter);
      const name = `${base}-head${faceLandmarks.length > 1 ? `-${i + 1}` : ''}.png`;
      const fig = document.createElement('figure');
      const btn = Object.assign(document.createElement('button'), {
        type: 'button', className: 'btn btn-primary', textContent: 'Download PNG',
      });
      btn.setAttribute('aria-label', `Download ${name}`);
      btn.addEventListener('click', () => canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        Object.assign(document.createElement('a'), { href: url, download: name }).click();
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }, 'image/png'));
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `Head ${i + 1} cut out`);
      fig.append(canvas, btn);
      $('.faces').append(fig);
    });
    img.close();
  } catch (err) {
    $('.card-summary').textContent = '';
    $('.card-error').textContent = err.message || String(err);
  }
}

const zone = document.getElementById('dropzone');
const input = document.getElementById('file-input');
document.getElementById('browse').addEventListener('click', () => input.click());
input.addEventListener('change', () => { [...input.files].forEach(process); input.value = ''; });
['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('is-over'); }));
['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, e => {
  e.preventDefault();
  if (ev === 'dragleave' && zone.contains(e.relatedTarget)) return;
  zone.classList.remove('is-over');
}));
zone.addEventListener('drop', e => [...e.dataTransfer.files].filter(f => f.type.startsWith('image/')).forEach(process));
document.addEventListener('paste', e => [...e.clipboardData.files].filter(f => f.type.startsWith('image/')).forEach(process));
document.getElementById('year').textContent = new Date().getFullYear();
