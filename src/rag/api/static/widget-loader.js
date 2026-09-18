(function () {
  // Derive our backend's origin from this script's own <script src>, so the
  // one-line embed snippet never has to hardcode a URL that could drift from
  // wherever the backend ends up actually hosted.
  var scriptEl = document.currentScript;
  if (!scriptEl || !scriptEl.src) return;
  var ORIGIN = new URL(scriptEl.src).origin;

  var BUBBLE_SIZE = 60;
  var PANEL_WIDTH = 380;
  var PANEL_HEIGHT = 600;

  var bubble = document.createElement('button');
  bubble.setAttribute('aria-label', 'Open chat assistant');
  bubble.textContent = '💬'; // speech balloon -- zero-dependency icon
  Object.assign(bubble.style, {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    width: BUBBLE_SIZE + 'px',
    height: BUBBLE_SIZE + 'px',
    borderRadius: '50%',
    border: 'none',
    background: '#2f6f4f',
    color: '#fff',
    fontSize: '26px',
    lineHeight: BUBBLE_SIZE + 'px',
    padding: '0',
    cursor: 'pointer',
    boxShadow: '0 4px 14px rgba(0,0,0,.25)',
    zIndex: 2147483000,
  });

  var frame = document.createElement('iframe');
  frame.src = ORIGIN + '/';
  frame.title = 'Chat assistant';
  Object.assign(frame.style, {
    position: 'fixed',
    bottom: (BUBBLE_SIZE + 32) + 'px',
    right: '20px',
    width: PANEL_WIDTH + 'px',
    height: PANEL_HEIGHT + 'px',
    maxWidth: 'calc(100vw - 32px)',
    maxHeight: 'calc(100vh - 120px)',
    border: 'none',
    borderRadius: '16px',
    boxShadow: '0 8px 30px rgba(0,0,0,.25)',
    zIndex: 2147483000 - 1,
    display: 'none',
    colorScheme: 'light dark',
  });

  var isOpen = false;
  function setOpen(next) {
    isOpen = next;
    frame.style.display = isOpen ? 'block' : 'none';
    bubble.textContent = isOpen ? '✕' : '💬';
    bubble.style.fontSize = isOpen ? '28px' : '26px';
    bubble.setAttribute('aria-label', isOpen ? 'Close chat assistant' : 'Open chat assistant');
  }

  bubble.addEventListener('click', function () {
    setOpen(!isOpen);
  });

  function mount() {
    document.body.appendChild(frame);
    document.body.appendChild(bubble);
  }

  if (document.body) {
    mount();
  } else {
    document.addEventListener('DOMContentLoaded', mount);
  }
})();
