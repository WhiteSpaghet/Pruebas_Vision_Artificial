/**
 * notifications.js
 * Lightweight toast notification helper for VISION_ARTIFICIAL
 *
 * Usage:
 *   Notifications.showToast("Procesando completado", { type: 'success', ttl: 3000 });
 *   const id = Notifications.showToast("Click to undo", { type: 'warning', action: { label: 'Undo', onClick: ()=>{} }});
 *   Notifications.dismissToast(id);
 *   Notifications.clearToasts();
 *
 * The script injects minimal CSS automatically.
 */

(function (global) {
  const DEFAULT_TTL = 4500;
  const MAX_TOASTS = 6; // maximum visible toasts, older ones removed
  const DEFAULT_POSITION = 'bottom-right'; // top-left, top-right, bottom-left, bottom-right
  const CSS_ID = 'va-notifications-css';
  const CONTAINER_ID = 'va-toast-container';

  // Simple unique id generator
  let idCounter = 0;
  function uid(prefix = 'va-toast-') {
    idCounter += 1;
    return prefix + Date.now().toString(36) + '-' + idCounter;
  }

  // Insert default CSS (only once)
  function injectCSS() {
    if (document.getElementById(CSS_ID)) return;
    const css = `
/* VISION_ARTIFICIAL notifications CSS */
#${CONTAINER_ID} {
  position: fixed;
  z-index: 99999;
  pointer-events: none;
  font-family: "Segoe UI", Roboto, Arial, sans-serif;
}
#${CONTAINER_ID}.top-left { top: 18px; left: 18px; }
#${CONTAINER_ID}.top-right { top: 18px; right: 18px; }
#${CONTAINER_ID}.bottom-left { bottom: 18px; left: 18px; }
#${CONTAINER_ID}.bottom-right { bottom: 18px; right: 18px; }

.va-toast {
  min-width: 260px;
  max-width: 420px;
  margin: 8px 0;
  background: rgba(30,30,30,0.95);
  color: #fff;
  padding: 12px 14px;
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
  display: flex;
  gap: 12px;
  align-items: flex-start;
  pointer-events: auto;
  opacity: 0;
  transform: translateY(10px);
  transition: opacity 260ms ease, transform 260ms ease;
  overflow: hidden;
}
.va-toast.show {
  opacity: 1;
  transform: translateY(0);
}
.va-toast .va-body {
  flex: 1 1 auto;
  min-width: 0;
}
.va-toast .va-title {
  font-weight: 600;
  margin-bottom: 4px;
  font-size: 0.95rem;
}
.va-toast .va-message {
  font-size: 0.90rem;
  line-height: 1.2;
}
.va-toast .va-actions {
  margin-left: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.va-toast button.va-action-btn {
  background: transparent;
  color: inherit;
  border: 1px solid rgba(255,255,255,0.12);
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}
.va-toast button.va-close-btn {
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  font-weight: 600;
  padding: 6px;
  font-size: 0.9rem;
}
.va-toast .va-progress {
  position: absolute;
  left: 0;
  bottom: 0;
  height: 4px;
  width: 100%;
  background: rgba(255,255,255,0.06);
}
.va-toast .va-progress > div {
  height: 100%;
  width: 100%;
  transform-origin: left center;
  transform: scaleX(1);
  transition: transform linear;
}

/* type variations */
.va-toast.type-info { background: linear-gradient(180deg, #2f3b74, #24305a); }
.va-toast.type-success { background: linear-gradient(180deg, #1b6b35, #145428); }
.va-toast.type-warning { background: linear-gradient(180deg, #8a5d1f, #6e4717); }
.va-toast.type-error { background: linear-gradient(180deg, #8a1f2c, #6b101a); }
/* text for light backgrounds fallback */
.va-toast.light { color: #111; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
`;
    const style = document.createElement('style');
    style.id = CSS_ID;
    style.appendChild(document.createTextNode(css));
    document.head.appendChild(style);
  }

  // Create container
  function getContainer(position = DEFAULT_POSITION) {
    injectCSS();
    let container = document.getElementById(CONTAINER_ID);
    if (!container) {
      container = document.createElement('div');
      container.id = CONTAINER_ID;
      container.className = position;
      document.body.appendChild(container);
    } else {
      // update position classes
      container.className = position;
    }
    return container;
  }

  // Create a toast element
  function createToastEl({ id, title, message, type, ttl, action }) {
    const wrapper = document.createElement('div');
    wrapper.className = `va-toast type-${type || 'info'}`;
    wrapper.setAttribute('role', 'status');
    wrapper.setAttribute('aria-live', 'polite');
    wrapper.setAttribute('data-toast-id', id);

    // body
    const body = document.createElement('div');
    body.className = 'va-body';

    if (title) {
      const t = document.createElement('div');
      t.className = 'va-title';
      t.innerText = title;
      body.appendChild(t);
    }

    if (message) {
      const m = document.createElement('div');
      m.className = 'va-message';
      m.innerText = message;
      body.appendChild(m);
    }

    // actions container
    const actions = document.createElement('div');
    actions.className = 'va-actions';

    // action button
    if (action && action.label && typeof action.onClick === 'function') {
      const actBtn = document.createElement('button');
      actBtn.className = 'va-action-btn';
      actBtn.type = 'button';
      actBtn.innerText = action.label;
      actBtn.addEventListener('click', (ev) => {
        try { action.onClick(ev); } catch (err) { console.error(err); }
      });
      actions.appendChild(actBtn);
    }

    // close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'va-close-btn';
    closeBtn.type = 'button';
    closeBtn.setAttribute('aria-label', 'Cerrar notificación');
    closeBtn.innerText = '✕';
    actions.appendChild(closeBtn);

    wrapper.appendChild(body);
    wrapper.appendChild(actions);

    // progress bar
    const progWrap = document.createElement('div');
    progWrap.className = 'va-progress';
    const prog = document.createElement('div');
    prog.style.background = 'linear-gradient(90deg, rgba(255,255,255,0.35), rgba(255,255,255,0.08))';
    progWrap.appendChild(prog);
    wrapper.appendChild(progWrap);

    return { wrapper, closeBtn, prog };
  }

  // internal store: id-> controllers
  const toasts = new Map();

  function showToast(message, opts = {}) {
    const {
      title = '',
      type = 'info',        // info | success | warning | error
      ttl = DEFAULT_TTL,
      position = DEFAULT_POSITION,
      action = null,        // { label: 'Undo', onClick: function }
      pauseOnHover = true,
      light = false         // alternate lighter style (for light backgrounds)
    } = opts;

    const container = getContainer(position);

    // keep number of toasts bounded
    if (container.children.length >= MAX_TOASTS) {
      // remove oldest (first child)
      const first = container.children[0];
      if (first) {
        const oldestId = first.getAttribute('data-toast-id');
        if (oldestId) dismissToast(oldestId);
      }
    }

    const id = uid();
    const { wrapper, closeBtn, prog } = createToastEl({ id, title, message, type, ttl, action });
    if (light) wrapper.classList.add('light');
    wrapper.dataset.ttl = ttl;

    // append and show
    container.appendChild(wrapper);
    // small delay for transition
    requestAnimationFrame(() => wrapper.classList.add('show'));

    // timer control
    let start = performance.now();
    let paused = false;
    let elapsedPaused = 0;
    let rafId = null;

    // animate progress using transform scaleX
    function startProgress() {
      const startTime = performance.now();
      function step(now) {
        if (paused) { rafId = requestAnimationFrame(step); return; }
        const elapsed = now - startTime;
        const ratio = Math.max(0, Math.min(1, (ttl - elapsed) / ttl)); // reverse scale
        prog.style.transform = `scaleX(${ratio})`;
        if (elapsed < ttl) {
          rafId = requestAnimationFrame(step);
        } else {
          // finished
          dismissToast(id);
        }
      }
      rafId = requestAnimationFrame(step);
    }

    // fallback timeout (in case raf timing issues)
    const fallbackTimeout = setTimeout(() => {
      dismissToast(id);
    }, ttl + 300); // small buffer

    // Pause on hover
    function onMouseEnter() {
      if (!pauseOnHover) return;
      paused = true;
      wrapper.classList.add('paused');
    }
    function onMouseLeave() {
      if (!pauseOnHover) return;
      paused = false;
      wrapper.classList.remove('paused');
    }
    if (pauseOnHover) {
      wrapper.addEventListener('mouseenter', onMouseEnter);
      wrapper.addEventListener('mouseleave', onMouseLeave);
    }

    // close button
    closeBtn.addEventListener('click', () => {
      dismissToast(id);
    });

    // store controller
    toasts.set(id, {
      id,
      el: wrapper,
      timeout: fallbackTimeout,
      rafId,
      startProgress,
      paused,
      cleanup: function () {
        // remove listeners
        if (pauseOnHover) {
          wrapper.removeEventListener('mouseenter', onMouseEnter);
          wrapper.removeEventListener('mouseleave', onMouseLeave);
        }
        closeBtn.removeEventListener('click', () => {});
      }
    });

    // Start the progress animation (slight delay to ensure DOM computed)
    setTimeout(() => {
      const ctrl = toasts.get(id);
      if (ctrl) ctrl.startProgress();
    }, 50);

    // Return id for possible programmatic dismissal
    return id;
  }

  function dismissToast(id) {
    const ctrl = toasts.get(id);
    if (!ctrl) return false;
    const el = ctrl.el;
    // Cancel fallback timeout
    try { clearTimeout(ctrl.timeout); } catch (e) {}
    // cancel RAF if any (not stored robustly)
    // animate out
    el.classList.remove('show');
    el.style.transition = 'opacity 180ms ease, transform 180ms ease';
    el.style.opacity = '0';
    el.style.transform = 'translateY(6px) scale(0.99)';
    setTimeout(() => {
      try {
        if (el.parentNode) el.parentNode.removeChild(el);
      } catch (e) {}
    }, 200);
    // cleanup
    if (typeof ctrl.cleanup === 'function') ctrl.cleanup();
    toasts.delete(id);
    return true;
  }

  function clearToasts() {
    for (const [id] of Array.from(toasts.entries())) {
      dismissToast(id);
    }
    // also remove container children forcibly if any left
    const container = document.getElementById(CONTAINER_ID);
    if (container) container.innerHTML = '';
    toasts.clear();
  }

  // Expose API
  const API = {
    showToast,
    dismissToast,
    clearToasts,
    // convenience helpers
    info: (msg, opts = {}) => showToast(msg, Object.assign({ type: 'info' }, opts)),
    success: (msg, opts = {}) => showToast(msg, Object.assign({ type: 'success' }, opts)),
    warning: (msg, opts = {}) => showToast(msg, Object.assign({ type: 'warning' }, opts)),
    error: (msg, opts = {}) => showToast(msg, Object.assign({ type: 'error' }, opts)),
    setPosition: (pos) => getContainer(pos) // will create container with new position class
  };

  // Attach to window for global use
  if (!global.Notifications) global.Notifications = API;
  else {
    // if already present, merge helpers
    Object.assign(global.Notifications, API);
  }
})(window);

/* ===========================
   Examples (use in console or other scripts)
   ===========================
   Notifications.showToast("Información general", { type: 'info' });
   Notifications.success("Guardado correctamente", { ttl: 2500 });
   Notifications.warning("Atención: tamaño grande", { action: { label: 'Continuar', onClick: ()=>{ console.log('seguir') } }});
   const id = Notifications.showToast("Puedes deshacer", { type: 'warning', ttl: 6000 });
   Notifications.dismissToast(id);
   Notifications.clearToasts();
   Notifications.setPosition('top-right'); // or 'top-left', 'bottom-left', 'bottom-right'
*/
