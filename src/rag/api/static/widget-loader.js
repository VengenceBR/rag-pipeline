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
  Object.assign(bubble.style, {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    width: BUBBLE_SIZE + 'px',
    height: BUBBLE_SIZE + 'px',
    borderRadius: '50%',
    border: 'none',
    background: '#ffffff',
    color: '#c0272f',
    fontSize: '26px',
    lineHeight: BUBBLE_SIZE + 'px',
    padding: '0',
    cursor: 'pointer',
    boxShadow: '0 4px 14px rgba(0,0,0,.25)',
    zIndex: 2147483000,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  });

  var bubbleIcon = document.createElement('img');
  bubbleIcon.src = ORIGIN + '/assets/corvit-mark.png';
  bubbleIcon.alt = '';
  Object.assign(bubbleIcon.style, {
    width: '62%',
    height: 'auto',
    display: 'block',
  });
  bubble.appendChild(bubbleIcon);

  var frame = document.createElement('iframe');
  // Cache-bust with a unique query string every time this script runs, so a
  // stale cached copy of "/" literally cannot be served -- the URL itself is
  // new, independent of whether the browser honors Cache-Control correctly.
  frame.src = ORIGIN + '/?v=' + Date.now();
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
    bubble.textContent = isOpen ? '✕' : ''; // escaped so it can't mojibake regardless of headers
    if (!isOpen) bubble.appendChild(bubbleIcon); // textContent='' above detached it
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
