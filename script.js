/**
 * The Lorax AI Voice Studio - Frontend Logic
 * MediaRecorder audio capture, Web Audio API waveform visualizer,
 * Direct RVC conversion with RMVPE pitch tracking.
 */

// ======================= GLOBAL STATE =======================
const STATE = {
  apiUrl: getInitialApiUrl(),
  selectedCharacter: 'lorax', // 'lorax' | 'pipsqueak'
  pitchShift: 0,
  recordedBlob: null,
  recordedAudioUrl: null,
  convertedAudioUrl: null,
  isRecording: false,
  recordStartTime: 0,
  timerInterval: null,
  mediaRecorder: null,
  audioChunks: [],
  audioContext: null,
  analyser: null,
  animationFrameId: null,
  stream: null,
  systemStatus: null,
};

// ======================= INITIALIZATION & NETWORKING =======================

function getInitialApiUrl() {
  const saved = localStorage.getItem('lorax_api_url');
  if (saved) return saved.replace(/\/+$/, '');
  if (window.location.protocol.startsWith('http')) {
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

document.addEventListener('DOMContentLoaded', () => {
  initDOM();
  initVisualizerCanvas();
  fetchSystemStatus();
  setInterval(fetchSystemStatus, 15000);
});

// ======================= DOM ELEMENTS =======================
let DOM = {};

function initDOM() {
  DOM = {
    // Header
    serverStatusBadge: document.getElementById('serverStatusBadge'),
    serverStatusText: document.getElementById('serverStatusText'),
    settingsBtn: document.getElementById('settingsBtn'),
    footerHardwareInfo: document.getElementById('footerHardwareInfo'),

    // Characters
    cardLorax: document.getElementById('card-lorax'),
    cardPipsqueak: document.getElementById('card-pipsqueak'),
    badgeLorax: document.getElementById('badge-lorax'),
    badgePipsqueak: document.getElementById('badge-pipsqueak'),
    hintLorax: document.getElementById('hint-lorax'),
    hintPipsqueak: document.getElementById('hint-pipsqueak'),

    // Recording
    startRecordBtn: document.getElementById('startRecordBtn'),
    stopRecordBtn: document.getElementById('stopRecordBtn'),
    recordingTimer: document.getElementById('recordingTimer'),
    visualizerHint: document.getElementById('visualizerHint'),
    waveformCanvas: document.getElementById('waveformCanvas'),
    originalAudioBox: document.getElementById('originalAudioBox'),
    originalAudioPlayer: document.getElementById('originalAudioPlayer'),
    originalDuration: document.getElementById('originalDuration'),

    // Synthesis
    convertBtn: document.getElementById('convertBtn'),
    pitchSlider: document.getElementById('pitchSlider'),
    pitchValDisplay: document.getElementById('pitchValDisplay'),
    resetPitchBtn: document.getElementById('resetPitchBtn'),
    loadingState: document.getElementById('loadingState'),
    loadingStageTitle: document.getElementById('loadingStageTitle'),
    loadingStageSub: document.getElementById('loadingStageSub'),
    progressBarFill: document.getElementById('progressBarFill'),

    // Result
    resultBox: document.getElementById('resultBox'),
    resultCharacterNotice: document.getElementById('resultCharacterNotice'),
    conversionDetails: document.getElementById('conversionDetails'),
    resultAudioPlayer: document.getElementById('resultAudioPlayer'),
    downloadAudioBtn: document.getElementById('downloadAudioBtn'),
    newRecordBtn: document.getElementById('newRecordBtn'),

    // Settings Modal
    settingsModal: document.getElementById('settingsModal'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    apiUrlInput: document.getElementById('apiUrlInput'),
    setLocalhostBtn: document.getElementById('setLocalhostBtn'),
    setSameOriginBtn: document.getElementById('setSameOriginBtn'),
    testConnectionBtn: document.getElementById('testConnectionBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    diagConnStatus: document.getElementById('diagConnStatus'),
    diagGpu: document.getElementById('diagGpu'),
    diagLorax: document.getElementById('diagLorax'),
    diagPipsqueak: document.getElementById('diagPipsqueak'),

    // Toasts
    toastContainer: document.getElementById('toastContainer')
  };

  bindEvents();
}

function bindEvents() {
  // Character selection
  DOM.cardLorax.addEventListener('click', () => selectCharacter('lorax'));
  DOM.cardPipsqueak.addEventListener('click', () => selectCharacter('pipsqueak'));

  // Recording buttons
  DOM.startRecordBtn.addEventListener('click', startRecording);
  DOM.stopRecordBtn.addEventListener('click', stopRecording);

  // Conversion & Reset
  DOM.convertBtn.addEventListener('click', convertVoice);
  DOM.newRecordBtn.addEventListener('click', resetForNewRecord);

  // Pitch shift preserves the source timing and pitch contour.
  if (DOM.pitchSlider) {
    DOM.pitchSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      STATE.pitchShift = val;
      updatePitchDisplay(val);
    });
  }
  if (DOM.resetPitchBtn) {
    DOM.resetPitchBtn.addEventListener('click', () => {
      STATE.pitchShift = 0;
      if (DOM.pitchSlider) DOM.pitchSlider.value = 0;
      updatePitchDisplay(0);
    });
  }

  // Settings
  DOM.settingsBtn.addEventListener('click', openSettingsModal);
  DOM.closeSettingsBtn.addEventListener('click', closeSettingsModal);
  DOM.settingsModal.addEventListener('click', (e) => {
    if (e.target === DOM.settingsModal) closeSettingsModal();
  });
  DOM.setLocalhostBtn.addEventListener('click', () => {
    DOM.apiUrlInput.value = 'http://127.0.0.1:8000';
  });
  DOM.setSameOriginBtn.addEventListener('click', () => {
    DOM.apiUrlInput.value = window.location.origin;
  });
  DOM.testConnectionBtn.addEventListener('click', testCustomApiConnection);
  DOM.saveSettingsBtn.addEventListener('click', saveApiSettings);

  // Quote Pills
  document.querySelectorAll('.phrase-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const text = pill.textContent.replace(/^"|"$/g, '');
      showToast(`Press "Start Recording" and say: "${text}"`, 'info', 4000);
    });
  });
}

// ======================= CHARACTERS =======================

function selectCharacter(charId) {
  STATE.selectedCharacter = charId;
  DOM.cardLorax.classList.toggle('active', charId === 'lorax');
  DOM.cardPipsqueak.classList.toggle('active', charId === 'pipsqueak');
}

// ======================= PITCH =======================

function updatePitchDisplay(val) {
  DOM.pitchValDisplay.textContent = `${val > 0 ? '+' : ''}${val} semitones${val === 0 ? ' (Original)' : ''}`;
}

// ======================= SERVER STATUS =======================

async function fetchSystemStatus() {
  try {
    const res = await fetch(`${STATE.apiUrl}/api/status`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    STATE.systemStatus = data;
    renderSystemStatus(data);
  } catch (err) {
    console.warn('Server unreachable:', err);
    setServerOfflineUI();
  }
}

function renderSystemStatus(data) {
  const dot = DOM.serverStatusBadge.querySelector('.status-dot');
  dot.className = 'status-dot online';

  if (data.hardware?.cuda_available) {
    DOM.serverStatusText.textContent = `Online (${data.hardware.gpu_name || 'CUDA'})`;
    DOM.serverStatusBadge.title = `RVC • CUDA: ${data.hardware.gpu_name}`;
  } else {
    DOM.serverStatusText.textContent = 'RVC on CPU';
    DOM.serverStatusBadge.title = 'Running on CPU';
  }

  DOM.footerHardwareInfo.textContent = data.hardware?.cuda_available
    ? `RVC + RMVPE • CUDA: ${data.hardware.gpu_name}`
    : 'RVC + RMVPE • CPU';

  const chars = data.characters || {};

  // The Lorax
  if (chars.lorax) {
    if (chars.lorax.available) {
      DOM.badgeLorax.textContent = 'READY';
      DOM.badgeLorax.className = 'char-badge-status ready';
    } else {
      DOM.badgeLorax.textContent = chars.lorax.training?.active ? 'Training Voice' : chars.lorax.has_model ? 'Setup Needed' : 'Model Needed';
      DOM.badgeLorax.className = 'char-badge-status missing';
    }
  }

  // Pipsqueak
  if (chars.pipsqueak) {
    if (chars.pipsqueak.available) {
      DOM.badgePipsqueak.textContent = 'READY';
      DOM.badgePipsqueak.className = 'char-badge-status ready';
    } else {
      DOM.badgePipsqueak.textContent = chars.pipsqueak.training?.active ? 'Training Voice' : chars.pipsqueak.has_model ? 'Setup Needed' : 'Model Needed';
      DOM.badgePipsqueak.className = 'char-badge-status missing';
    }
  }
  for (const [id, hint] of [['lorax', DOM.hintLorax], ['pipsqueak', DOM.hintPipsqueak]]) {
    const char = chars[id];
    hint.textContent = !char?.available && char?.training?.active
      ? (char.training.current_epoch ? `Training voice • Epoch ${char.training.current_epoch}/${char.training.target_epochs}` : 'Preparing your character voice...')
      : char?.available
      ? (char.has_index ? 'Voice model + retrieval index loaded' : 'Voice model loaded • retrieval off')
      : (char?.has_model ? 'RVC setup needed' : 'Trained voice model needed');
  }
}

function setServerOfflineUI() {
  const dot = DOM.serverStatusBadge.querySelector('.status-dot');
  dot.className = 'status-dot offline';
  DOM.serverStatusText.textContent = 'Server Offline (Run server.py)';
  DOM.footerHardwareInfo.textContent = 'Server not responding at: ' + STATE.apiUrl;

  DOM.badgeLorax.textContent = 'Offline';
  DOM.badgeLorax.className = 'char-badge-status';
  DOM.badgePipsqueak.textContent = 'Offline';
  DOM.badgePipsqueak.className = 'char-badge-status';
}

// ======================= AUDIO RECORDING (MediaRecorder) =======================

async function startRecording() {
  try {
    STATE.audioChunks = [];
    STATE.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 44100,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      }
    });

    let mimeType = 'audio/webm';
    if (!MediaRecorder.isTypeSupported('audio/webm')) {
      if (MediaRecorder.isTypeSupported('audio/ogg')) mimeType = 'audio/ogg';
      else mimeType = '';
    }

    const options = mimeType ? { mimeType } : {};
    STATE.mediaRecorder = new MediaRecorder(STATE.stream, options);

    STATE.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        STATE.audioChunks.push(event.data);
      }
    };

    STATE.mediaRecorder.onstop = onRecordingStop;

    STATE.mediaRecorder.start(100);
    STATE.isRecording = true;
    STATE.recordStartTime = Date.now();
    setupAudioAnalyser(STATE.stream);

    // UI Updates
    DOM.startRecordBtn.disabled = true;
    DOM.stopRecordBtn.disabled = false;
    DOM.convertBtn.disabled = true;
    DOM.visualizerHint.textContent = 'Listening... Speak into your microphone!';
    DOM.waveformCanvas.classList.add('recording-active');

    STATE.timerInterval = setInterval(updateRecordingTimer, 100);
    showToast('Recording started... Speak into your microphone!', 'info', 2500);

  } catch (err) {
    console.error('Microphone access error:', err);
    showToast(
      'Unable to access microphone. Please enable microphone permissions in your browser settings.',
      'error',
      6000
    );
  }
}

function updateRecordingTimer() {
  const elapsedMs = Date.now() - STATE.recordStartTime;
  const totalSeconds = Math.floor(elapsedMs / 1000);
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const ss = String(totalSeconds % 60).padStart(2, '0');
  DOM.recordingTimer.textContent = `${mm}:${ss}`;

  // Limit recording to 30 seconds
  if (totalSeconds >= 30) {
    stopRecording();
    showToast('30-second maximum reached for optimal processing.', 'warning', 4000);
  }
}

function stopRecording() {
  if (!STATE.isRecording) return;
  STATE.isRecording = false;

  clearInterval(STATE.timerInterval);
  DOM.startRecordBtn.disabled = false;
  DOM.stopRecordBtn.disabled = true;
  DOM.waveformCanvas.classList.remove('recording-active');
  DOM.visualizerHint.textContent = 'Recording stopped';

  if (STATE.mediaRecorder && STATE.mediaRecorder.state !== 'inactive') {
    STATE.mediaRecorder.stop();
  }

  if (STATE.stream) {
    STATE.stream.getTracks().forEach(t => t.stop());
  }
}

function onRecordingStop() {
  STATE.recordedBlob = new Blob(STATE.audioChunks, {
    type: STATE.mediaRecorder?.mimeType || 'audio/webm'
  });

  if (STATE.recordedAudioUrl) {
    URL.revokeObjectURL(STATE.recordedAudioUrl);
  }
  STATE.recordedAudioUrl = URL.createObjectURL(STATE.recordedBlob);

  DOM.originalAudioPlayer.src = STATE.recordedAudioUrl;
  DOM.originalAudioBox.style.display = 'block';

  const durationSec = Math.max(1, Math.round((Date.now() - STATE.recordStartTime) / 1000));
  DOM.originalDuration.textContent = `${durationSec}s`;

  DOM.convertBtn.disabled = false;
  showToast('Recording captured! Now click "Convert Voice with RVC".', 'info', 3500);
}

// ======================= VISUALIZER (Web Audio API) =======================

function initVisualizerCanvas() {
  const canvas = DOM.waveformCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  drawIdleWave(canvas, ctx);
}

function setupAudioAnalyser(stream) {
  try {
    if (STATE.audioContext) STATE.audioContext.close().catch(() => {});
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    STATE.audioContext = new AudioCtx();
    const source = STATE.audioContext.createMediaStreamSource(stream);
    STATE.analyser = STATE.audioContext.createAnalyser();
    STATE.analyser.fftSize = 256;
    source.connect(STATE.analyser);

    startVisualizerLoop();
  } catch (e) {
    console.warn('Web Audio API not supported:', e);
  }
}

function startVisualizerLoop() {
  const canvas = DOM.waveformCanvas;
  const ctx = canvas.getContext('2d');
  const bufferLength = STATE.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!STATE.isRecording) {
      drawIdleWave(canvas, ctx);
      return;
    }

    STATE.animationFrameId = requestAnimationFrame(draw);
    STATE.analyser.getByteFrequencyData(dataArray);

    ctx.fillStyle = '#09150e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height * 0.95;

      const grad = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight);
      grad.addColorStop(0, '#ffcb2b');
      grad.addColorStop(0.5, '#ff7a18');
      grad.addColorStop(1, '#70c436');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, canvas.height - barHeight, barWidth - 2, barHeight, [4, 4, 0, 0]);
      ctx.fill();

      x += barWidth;
    }
  }

  draw();
}

function drawIdleWave(canvas, ctx) {
  ctx.fillStyle = '#09150e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = 'rgba(112, 196, 54, 0.25)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const midY = canvas.height / 2;
  ctx.moveTo(0, midY);
  ctx.lineTo(canvas.width, midY);
  ctx.stroke();
}

// ======================= RVC CONVERSION =======================

async function convertVoice() {
  if (!STATE.recordedBlob) {
    showToast('Please record your voice with the microphone first!', 'warning', 3000);
    return;
  }

  // Check character availability
  if (STATE.systemStatus && STATE.systemStatus.characters) {
    const charInfo = STATE.systemStatus.characters[STATE.selectedCharacter];
    if (charInfo && !charInfo.available) {
      showMissingReferenceAlert(STATE.selectedCharacter, charInfo);
      return;
    }
  }

  DOM.convertBtn.disabled = true;
  DOM.loadingState.style.display = 'flex';
  DOM.resultBox.style.display = 'none';

  let stageTimer = runInferenceProgressStages();

  const formData = new FormData();
  formData.append('audio', STATE.recordedBlob, 'voice_input.webm');
  formData.append('character', STATE.selectedCharacter);

  formData.append('pitch_shift', STATE.pitchShift);
  formData.append('rms_mix_rate', 0);

  try {
    const res = await fetch(`${STATE.apiUrl}/api/convert`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(stageTimer);

    if (!res.ok) {
      let errDetail = 'Error calling server';
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || JSON.stringify(errJson);
      } catch (e) {
        errDetail = `Server returned HTTP ${res.status}`;
      }

      if (res.status === 404 || res.status === 400) {
        alert(`⚠️ Notice:\n\n${errDetail}`);
      } else {
        showToast(`Error (${res.status}): ${errDetail}`, 'error', 7000);
      }
      return;
    }

    const data = await res.json();

    DOM.conversionDetails.textContent = `RVC + RMVPE • Pitch: ${data.pitch_shift} semitones • Original speaking pace • Retrieval: ${data.index_rate > 0 ? 'on' : 'off'}`;

    // Fill audio player & download link
    const fullAudioUrl = `${STATE.apiUrl}${data.audio_url}`;
    DOM.resultAudioPlayer.src = fullAudioUrl;
    DOM.downloadAudioBtn.href = fullAudioUrl;
    DOM.downloadAudioBtn.download = `${data.character}_cloned_voice.wav`;

    DOM.resultCharacterNotice.textContent = data.character === 'lorax'
      ? 'Speaking as The Lorax — Guardian of the Forest'
      : 'Speaking as Pipsqueak — Bar-ba-loot Bear Cub';

    DOM.resultBox.style.display = 'flex';
    showToast('✨ Voice converted with RVC!', 'info', 4000);

    setTimeout(() => {
      DOM.resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      DOM.resultAudioPlayer.play().catch(() => {});
    }, 300);

  } catch (err) {
    clearInterval(stageTimer);
    console.error('Connection failure:', err);
    showToast(
      `Unable to reach backend at ${STATE.apiUrl}. Please verify server.py is running.`,
      'error',
      6000
    );
  } finally {
    DOM.convertBtn.disabled = false;
    DOM.loadingState.style.display = 'none';
  }
}

function runInferenceProgressStages() {
  const stages = [
    { title: 'Converting with RVC + RMVPE...', sub: 'Processing your recording; the first conversion also loads the models.', pct: '35%' },
    { title: 'RVC conversion in progress...', sub: 'Please wait. Processing time depends on the recording and your hardware.', pct: '65%' },
  ];

  let idx = 0;
  DOM.progressBarFill.style.width = stages[0].pct;
  DOM.loadingStageTitle.textContent = stages[0].title;
  DOM.loadingStageSub.textContent = stages[0].sub;

  const timer = setInterval(() => {
    idx++;
    if (idx < stages.length) {
      DOM.progressBarFill.style.width = stages[idx].pct;
      DOM.loadingStageTitle.textContent = stages[idx].title;
      DOM.loadingStageSub.textContent = stages[idx].sub;
    }
  }, 3500);

  return timer;
}

function showMissingReferenceAlert(characterId, charInfo) {
  const charName = characterId === 'lorax' ? 'The Lorax' : 'Pipsqueak';
  if (charInfo.training?.active) {
    showToast(`${charName}'s RVC voice is being trained. It will become available automatically when training finishes.`, 'info', 6000);
    return;
  }
  if (!charInfo.has_model) {
    alert(`RVC model needed for ${charName}.\n\nAdd a trained voice.pth to models/${characterId}/. Add its matching .index if available.\n\nA WAV sample alone is not an RVC model. See README.md for training instructions.`);
  } else {
    alert('RVC setup is incomplete. Run setup_rvc.ps1, then restart the server.');
  }
}

function resetForNewRecord() {
  DOM.resultBox.style.display = 'none';
  DOM.originalAudioBox.style.display = 'none';
  DOM.recordingTimer.textContent = '00:00';
  DOM.visualizerHint.textContent = 'Click "Start Recording" and speak clearly';
  DOM.convertBtn.disabled = true;
  STATE.recordedBlob = null;
  showToast('Ready for a new recording! Click "Start Recording".', 'info', 2500);
}

// ======================= SETTINGS MODAL =======================

function openSettingsModal() {
  DOM.apiUrlInput.value = STATE.apiUrl;
  testCustomApiConnection();
  DOM.settingsModal.style.display = 'flex';
}

function closeSettingsModal() {
  DOM.settingsModal.style.display = 'none';
}

async function testCustomApiConnection() {
  const url = (DOM.apiUrlInput.value || '').trim().replace(/\/+$/, '');
  DOM.diagConnStatus.textContent = 'Checking connection...';

  try {
    const res = await fetch(`${url}/api/status`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    DOM.diagConnStatus.textContent = 'Connected successfully! ✅';
    DOM.diagGpu.textContent = data.hardware?.cuda_available
      ? `CUDA Active (${data.hardware.gpu_name})`
      : 'CPU Mode';

    DOM.diagLorax.textContent = data.characters?.lorax?.available ? 'RVC ready' : data.characters?.lorax?.training?.active ? 'Training voice...' : data.characters?.lorax?.has_model ? 'Setup needed' : 'Trained model needed';
    DOM.diagPipsqueak.textContent = data.characters?.pipsqueak?.available ? 'RVC ready' : data.characters?.pipsqueak?.has_model ? 'Setup needed' : 'Trained model needed';

  } catch (err) {
    DOM.diagConnStatus.textContent = 'Connection failed ❌';
    DOM.diagGpu.textContent = 'N/A';
    DOM.diagLorax.textContent = 'N/A';
    DOM.diagPipsqueak.textContent = 'N/A';
  }
}

function saveApiSettings() {
  const newUrl = (DOM.apiUrlInput.value || '').trim().replace(/\/+$/, '');
  if (!newUrl) {
    showToast('Please enter a valid server URL!', 'warning', 3000);
    return;
  }
  STATE.apiUrl = newUrl;
  localStorage.setItem('lorax_api_url', newUrl);
  closeSettingsModal();
  fetchSystemStatus();
  showToast(`Server URL saved: ${newUrl}`, 'info', 3000);
}

// ======================= TOAST NOTIFICATIONS =======================

function showToast(message, type = 'info', duration = 3500) {
  const container = DOM.toastContainer;
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast-item toast-${type}`;

  const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : '🍃';
  const iconEl = document.createElement('span');
  iconEl.className = 'toast-icon';
  iconEl.textContent = icon;
  const messageEl = document.createElement('span');
  messageEl.className = 'toast-msg';
  messageEl.textContent = message;
  toast.append(iconEl, messageEl);

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}
