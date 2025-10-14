/**
 * dashboard.js
 * VISION_ARTIFICIAL
 *
 * Funcionalidades:
 * - Selector de tema con persistencia en localStorage y (si hay usuario) en Firestore
 * - Detecta estado de autenticación Firebase y actualiza UI
 * - Carga métricas (accuracy/loss) desde Firestore o REST API (/api/metrics)
 * - Carga historial de predicciones (filtrable por clase/confianza)
 * - Subida batch de imágenes para inferencia a /api/predict (o Cloud Function)
 * - Toggle para ensemble dinámico (invoca /api/ensemble/toggle)
 * - Toaster simple para notificaciones
 *
 * Dependencias front: Firebase (opcional, v9 modular recommended) si quieres auth/DB en frontend.
 * Si no usas Firebase en frontend, éste fallará a la hora de hacer onAuthStateChanged; en ese
 * caso, el script usa los endpoints REST del backend Flask (si están implementados).
 *
 * Elementos HTML esperados (IDs):
 * - #theme-select (select)
 * - #user-display (span/div)
 * - #login-btn, #logout-btn
 * - #metric-accuracy, #metric-loss (spans o divs)
 * - #history-table (tbody o container)
 * - #history-filter-class (select) , #history-filter-confidence (input range)
 * - #upload-form (form with file input name="files")
 * - #ensemble-toggle (checkbox/button)
 * - #toast-container (div)
 *
 * Ajusta nombres/IDs en el HTML si es necesario.
 */

/* ========== CONFIG ========== */
const CONFIG = {
  firebaseEnabled: true, // si usas Firebase en frontend pon true; si no, false
  firebaseConfigFile: '/static/firebase_config.json', // opcional, si cargas config via fetch
  api: {
    metrics: '/api/metrics',          // GET -> {accuracy:.., loss:.., timestamp:..}
    history: '/api/history',          // GET -> [{username, prediction, confidence, timestamp, image_url},...]
    predict: '/api/predict',          // POST (form-data, files[]) -> batch predictions
    ensembleToggle: '/api/ensemble/toggle' // POST {enable: true/false}
  },
  defaultTheme: 'theme_light_green.css',
  themePath: '/static/css/'          // ruta relativa desde dashboard.html
};

/* ========== Firebase (modular v9) placeholders ========== */
/* Si usas Firebase en frontend, importa y configura en tu template con:
   <script type="module">
     import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
     import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
     import { getFirestore, doc, getDoc, setDoc, onSnapshot, collection, query, orderBy, limit } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
     // then window.firebaseApp = initializeApp({...});
   </script>
*/
let firebaseApp = null;
let firebaseAuth = null;
let firebaseDb = null;

/* ========== Util: Toaster ========== */
function ensureToastContainer() {
  let c = document.getElementById('toast-container');
  if (!c) {
    c = document.createElement('div');
    c.id = 'toast-container';
    c.style.position = 'fixed';
    c.style.right = '20px';
    c.style.bottom = '20px';
    c.style.zIndex = '9999';
    document.body.appendChild(c);
  }
  return c;
}
function showToast(message, type = 'info', ttl = 4500) {
  const container = ensureToastContainer();
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.style.background = (type === 'error') ? '#ff6b6b' : (type === 'success') ? '#4caf50' : '#333';
  el.style.color = '#fff';
  el.style.padding = '10px 14px';
  el.style.marginTop = '8px';
  el.style.borderRadius = '6px';
  el.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
  el.innerText = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 300ms ease';
    el.style.opacity = '0';
    setTimeout(() => container.removeChild(el), 350);
  }, ttl);
}

/* ========== THEME MANAGEMENT ========== */
function applyThemeFile(themeFileName) {
  const linkId = 'theme-style';
  let linkEl = document.getElementById(linkId);
  if (!linkEl) {
    // crear enlace si no existe (esperamos <link id="theme-style" ...> en base.html)
    linkEl = document.createElement('link');
    linkEl.id = linkId;
    linkEl.rel = 'stylesheet';
    document.head.appendChild(linkEl);
  }
  linkEl.href = `${CONFIG.themePath}${themeFileName}`;
  localStorage.setItem('vision_theme', themeFileName);
}
function setupThemeSelector() {
  const sel = document.getElementById('theme-select');
  if (!sel) return;

  // Poblamos select (si no está hecho en HTML)
  if (sel.options.length === 0) {
    const themes = [
      {value: 'theme_light_green.css', label: 'Verde claro'},
      {value: 'theme_light_dark.css', label: 'Blanco & Negro'},
      {value: 'theme_dark_gray.css', label: 'Oscuro gris'}
    ];
    themes.forEach(t => {
      const o = document.createElement('option');
      o.value = t.value; o.text = t.label;
      sel.appendChild(o);
    });
  }

  // cargar preferencia: localStorage -> (si usuario logueado, luego se sobrescribe con DB)
  const localTheme = localStorage.getItem('vision_theme') || CONFIG.defaultTheme;
  sel.value = localTheme;
  applyThemeFile(localTheme);

  sel.addEventListener('change', async (e) => {
    const chosen = e.target.value;
    applyThemeFile(chosen);
    // si Firebase activo y user logueado, guardar preferencia en Firestore
    if (firebaseAuth && firebaseAuth.currentUser && firebaseDb) {
      try {
        const uid = firebaseAuth.currentUser.uid;
        // Firestore modular example (if loaded globally)
        if (window.firebase && window.firebase.firestore) {
          // legacy firebase global
          const docRef = window.firebase.firestore().collection('users').doc(uid);
          await docRef.set({theme: chosen}, {merge: true});
        } else if (typeof window.firebaseSetUserTheme === 'function') {
          // si implementas una función wrapper en template
          await window.firebaseSetUserTheme(uid, chosen);
        }
      } catch (err) {
        console.warn('No se pudo guardar theme en Firestore:', err);
      }
    }
    showToast('Tema cambiado', 'success', 1500);
  });
}

/* ========== AUTH STATE & UI UPDATE ========== */
function setUserUI(user) {
  const userDisplay = document.getElementById('user-display');
  const loginBtn = document.getElementById('login-btn');
  const logoutBtn = document.getElementById('logout-btn');
  if (user) {
    if (userDisplay) userDisplay.innerText = user.displayName || user.email || user.uid;
    if (loginBtn) loginBtn.style.display = 'none';
    if (logoutBtn) logoutBtn.style.display = 'inline-block';
    // load user-specific prefs if exist
    loadUserThemePreference(user);
  } else {
    if (userDisplay) userDisplay.innerText = 'Invitado';
    if (loginBtn) loginBtn.style.display = 'inline-block';
    if (logoutBtn) logoutBtn.style.display = 'none';
  }
}
async function loadUserThemePreference(user) {
  if (!user) return;
  // Try Firestore first (if available)
  try {
    if (window.firebase && window.firebase.firestore) {
      const docRef = window.firebase.firestore().collection('users').doc(user.uid);
      const docSnap = await docRef.get();
      if (docSnap.exists) {
        const data = docSnap.data();
        if (data && data.theme) {
          const sel = document.getElementById('theme-select');
          if (sel) { sel.value = data.theme; applyThemeFile(data.theme); }
        }
      }
    }
  } catch (err) {
    console.warn('No se pudo leer preferencia de tema en Firestore:', err);
  }
}

/* ========== METRICS / DASHBOARD ========== */
async function fetchMetricsREST() {
  try {
    const res = await fetch(CONFIG.api.metrics);
    if (!res.ok) throw new Error('Error fetch metrics');
    return await res.json(); // expected {accuracy:0.83, loss:0.45, ...}
  } catch (err) {
    console.warn('fetchMetricsREST failed', err);
    return null;
  }
}
async function fetchMetricsFirestoreRealtime() {
  // If Firestore is available, set up onSnapshot (real-time)
  if (window.firebase && window.firebase.firestore) {
    try {
      const col = window.firebase.firestore().collection('metrics').orderBy('timestamp','desc').limit(1);
      col.onSnapshot(snap => {
        snap.forEach(doc => {
          const data = doc.data();
          updateMetricsUI(data);
        });
      });
      return true;
    } catch (err) {
      console.warn('Firestore realtime metrics failed:', err);
      return false;
    }
  }
  return false;
}
function updateMetricsUI(data) {
  if (!data) return;
  const accEl = document.getElementById('metric-accuracy');
  const lossEl = document.getElementById('metric-loss');
  if (accEl) accEl.innerText = (data.accuracy !== undefined) ? ( (data.accuracy*100).toFixed(2) + '%' ) : 'N/A';
  if (lossEl) lossEl.innerText = (data.loss !== undefined) ? data.loss.toFixed(4) : 'N/A';
}
async function refreshMetrics() {
  // prefer firestore realtime if configured
  const realtimeBound = await fetchMetricsFirestoreRealtime();
  if (!realtimeBound) {
    // fallback to REST polling
    const data = await fetchMetricsREST();
    if (data) updateMetricsUI(data);
  }
}

/* ========== HISTORY (PREDS) ========== */
async function fetchHistoryREST() {
  try {
    const res = await fetch(CONFIG.api.history);
    if (!res.ok) throw new Error('fetch history error');
    return await res.json(); // expected array
  } catch (err) {
    console.warn('fetchHistoryREST failed', err);
    return [];
  }
}
function renderHistory(items, filter = {}) {
  const tbody = document.getElementById('history-table');
  if (!tbody) return;
  tbody.innerHTML = '';
  const classFilter = filter.classIndex;
  const minConf = (filter.minConfidence !== undefined) ? filter.minConfidence : 0;
  items.forEach(it => {
    if (classFilter !== undefined && it.prediction_index !== undefined && +it.prediction_index !== +classFilter) return;
    if (it.confidence !== undefined && +it.confidence < +minConf) return;
    const tr = document.createElement('tr');

    const tdTime = document.createElement('td');
    tdTime.innerText = it.timestamp ? (new Date(it.timestamp).toLocaleString()) : '-';
    tr.appendChild(tdTime);

    const tdUser = document.createElement('td');
    tdUser.innerText = it.username || it.user || 'anon';
    tr.appendChild(tdUser);

    const tdPred = document.createElement('td');
    tdPred.innerText = it.prediction || it.prediction_label || 'N/A';
    tr.appendChild(tdPred);

    const tdConf = document.createElement('td');
    tdConf.innerText = (it.confidence !== undefined) ? ( (it.confidence*100).toFixed(2) + '%' ) : '-';
    tr.appendChild(tdConf);

    const tdImg = document.createElement('td');
    if (it.image_url) {
      const img = document.createElement('img');
      img.src = it.image_url;
      img.style.width = '64px';
      img.style.height = '64px';
      img.style.objectFit = 'cover';
      tdImg.appendChild(img);
    } else tdImg.innerText = '-';
    tr.appendChild(tdImg);

    tbody.appendChild(tr);
  });
}
async function refreshHistory() {
  const items = await fetchHistoryREST();
  renderHistory(items, getHistoryFilterFromUI());
}
function getHistoryFilterFromUI() {
  const classSel = document.getElementById('history-filter-class');
  const confInp = document.getElementById('history-filter-confidence');
  const filter = {};
  if (classSel && classSel.value && classSel.value !== 'all') filter.classIndex = classSel.value;
  if (confInp) filter.minConfidence = parseFloat(confInp.value) / 100.0;
  return filter;
}
function setupHistoryFilters() {
  const classSel = document.getElementById('history-filter-class');
  const confInp = document.getElementById('history-filter-confidence');
  if (classSel) classSel.addEventListener('change', () => refreshHistory());
  if (confInp) confInp.addEventListener('input', () => refreshHistory());
}

/* ========== BATCH UPLOAD & PREDICTION ========== */
function setupUploadForm() {
  const form = document.getElementById('upload-form');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const filesInput = form.querySelector('input[type="file"][name="files"]');
    if (!filesInput || !filesInput.files || filesInput.files.length === 0) {
      showToast('Selecciona al menos una imagen', 'error');
      return;
    }
    const files = Array.from(filesInput.files);
    // Build FormData
    const fd = new FormData();
    files.forEach((f, i) => fd.append('files', f));
    // optional: include user token if Fire auth used
    try {
      showToast(`Procesando ${files.length} imagen(es)...`, 'info', 5000);
      const resp = await fetch(CONFIG.api.predict, {
        method: 'POST',
        body: fd
      });
      if (!resp.ok) {
        const txt = await resp.text();
        showToast('Error en la petición: ' + txt, 'error', 6000);
        return;
      }
      const result = await resp.json(); // expected array of predictions
      // render results (simple)
      if (result && Array.isArray(result)) {
        const out = document.getElementById('upload-results');
        if (out) {
          out.innerHTML = '';
          result.forEach(r => {
            const div = document.createElement('div');
            div.className = 'card';
            div.innerHTML = `<strong>${r.prediction}</strong> (${(r.confidence*100).toFixed(2)}%)`;
            out.appendChild(div);
          });
        }
        showToast('Predicciones completadas', 'success', 3000);
        // refresh history dashboard
        setTimeout(() => refreshHistory(), 800);
      }
    } catch (err) {
      console.error(err);
      showToast('Error procesando imágenes', 'error', 4000);
    }
  });
}

/* ========== ENSEMBLE TOGGLE ========== */
function setupEnsembleToggle() {
  const el = document.getElementById('ensemble-toggle');
  if (!el) return;
  el.addEventListener('change', async (e) => {
    const enable = !!e.target.checked;
    try {
      const res = await fetch(CONFIG.api.ensembleToggle, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable })
      });
      if (!res.ok) throw new Error('ensemble toggle failed');
      showToast(`Ensemble ${enable ? 'activado' : 'desactivado'}`, 'success', 1800);
    } catch (err) {
      console.warn(err);
      showToast('No se pudo cambiar ensemble', 'error', 2500);
      // revert UI
      e.target.checked = !enable;
    }
  });
}

/* ========== INIT DASHBOARD ========== */
async function initDashboard() {
  // Theme selector
  setupThemeSelector();

  // Setup filters & upload
  setupHistoryFilters();
  setupUploadForm();
  setupEnsembleToggle();

  // Auth state management (if firebase loaded globally)
  try {
    if (window.firebase && window.firebase.auth) {
      // legacy firebase global object
      window.firebase.auth().onAuthStateChanged(user => {
        setUserUI(user);
      });
    } else if (window.getFirebaseAuth) {
      // optional wrapper: getFirebaseAuth should return auth instance and set up listener
      const auth = window.getFirebaseAuth();
      auth.onAuthStateChanged(user => setUserUI(user));
    } else {
      // fallback: try to call backend endpoint to get session state
      const resp = await fetch('/api/whoami');
      if (resp.ok) {
        const u = await resp.json(); // {username:..., email:...} or {anonymous:true}
        if (u && u.username) setUserUI(u);
      }
    }
  } catch (err) {
    console.warn('Auth detection failed:', err);
  }

  // First load of metrics & history
  await refreshMetrics();
  await refreshHistory();

  // Optionally refresh periodically
  setInterval(async () => {
    await refreshMetrics();
  }, 30_000); // actualizar métricas cada 30s

  // small UI polish
  document.body.classList.add('dashboard-ready');
  showToast('Dashboard listo', 'success', 1200);
}

/* Auto-init cuando DOM esté listo */
document.addEventListener('DOMContentLoaded', () => {
  initDashboard().catch(err => {
    console.error('initDashboard error', err);
    showToast('Error inicializando dashboard', 'error', 4000);
  });
});
