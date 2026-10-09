/* CogniVault UI — app.js
 * Vanilla ES2020, no modules/build, no external deps. Same-origin /api/* only.
 * Sections: state · dom · utils · api · sse parser · markdown · toasts ·
 *           status · theme · settings · upload · env · chat · history · init
 */
(function () {
  "use strict";

  /* ============================ STATE ============================ */
  const state = {
    config: null,          // full config object from GET /api/config
    chatId: null,          // current conversation id (null = fresh)
    messages: [],          // visible turns: { role, content, sources?, rag? }
    rag: true,             // RAG toggle
    streaming: false,      // a chat request is in flight
    abort: null,           // AbortController for current chat stream
    tokenVisible: false,   // masked token field state
    pipTokenVisible: false, // masked SberOSC token field state
    loginTokenVisible: false, // masked login-modal token field state
    confConfig: null,      // last GET /api/confluence/config
    confAuthMode: "basic", // "basic" | "pat"
    confPasswordVisible: false,
    confPatVisible: false,
    confSyncing: false,    // a confluence sync SSE is in flight
    collection: null,      // last successful GET /api/admin/collection
    reindexJob: null,      // jobId of the vault reindex we are watching
    reindexTimer: null,    // setTimeout handle of the reindex poll loop
    evalTimer: null,       // то же для опроса прогона оценки
    reindexBusy: false,    // a start request is in flight (double-click guard)
    reindexWatching: false, // we saw this job running → announce its result
    rebuildJob: null,      // jobId of the collection rebuild we are watching
    rebuildTimer: null,
    rebuildBusy: false,
    rebuildWatching: false,
    rebuildPhase: null,    // last phase logged, so transitions log once
  };

  /* ============================ DOM ============================ */
  const $ = (id) => document.getElementById(id);
  const dom = {
    // header / status
    pillCogni: $("pill-cognivault"),
    pillCert: $("pill-cert"),
    pillEnv: $("pill-env"),
    themeBtn: $("theme-btn"),
    settingsBtn: $("settings-btn"),
    // identity (server mode)
    identity: $("identity"),
    identityUser: $("identity-user"),
    logoutBtn: $("logout-btn"),
    // rail
    newchatBtn: $("newchat-btn"),
    chatlist: $("chatlist"),
    historyCount: $("history-count"),
    // banner
    banner: $("setup-banner"),
    bannerText: $("setup-banner-text"),
    bannerBtn: $("setup-banner-btn"),
    // chat
    stream: $("stream"),
    thread: $("thread"),
    ragSwitch: $("rag-switch"),
    ragSublabel: $("rag-sublabel"),
    input: $("composer-input"),
    sendBtn: $("send-btn"),
    // drawer
    scrim: $("scrim"),
    drawer: $("drawer"),
    drawerClose: $("drawer-close"),
    // settings §1
    cfgCognivaultUrl: $("cfg-cognivault-url"),
    cfgToken: $("cfg-token"),
    tokenToggle: $("token-toggle"),
    saveConn: $("save-conn"),
    savedConn: $("saved-conn"),
    // settings §2
    cfgGigaUrl: $("cfg-giga-url"),
    cfgModel: $("cfg-model"),
    cfgCert: $("cfg-cert"),
    cfgKey: $("cfg-key"),
    cfgPassphrase: $("cfg-passphrase"),
    cfgCa: $("cfg-ca"),
    cfgVerifySsl: $("cfg-verify-ssl"),
    saveCert: $("save-cert"),
    savedCert: $("saved-cert"),
    // §3 upload
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    uploadResult: $("upload-result"),
    uploadFilelist: $("upload-filelist"),
    uploadFiles: $("upload-files"),
    // §4 env
    cfgPipIndex: $("cfg-pip-index"),
    cfgPipToken: $("cfg-pip-token"),
    pipTokenToggle: $("pip-token-toggle"),
    saveEnvMirror: $("save-env-mirror"),
    savedEnvMirror: $("saved-env-mirror"),
    envSetupBtn: $("env-setup-btn"),
    envImportPath: $("env-import-path"),
    envImportBtn: $("env-import-btn"),
    envImportResult: $("env-import-result"),
    console: $("console"),
    // §3.5 confluence source
    sectionConfluence: $("section-confluence"),
    confAuthBasic: $("conf-auth-basic"),
    confAuthPat: $("conf-auth-pat"),
    confBasicFields: $("conf-basic-fields"),
    confPatFields: $("conf-pat-fields"),
    confLogin: $("conf-login"),
    confPassword: $("conf-password"),
    confPasswordToggle: $("conf-password-toggle"),
    confPasswordSaved: $("conf-password-saved"),
    confPat: $("conf-pat"),
    confPatToggle: $("conf-pat-toggle"),
    confPatSaved: $("conf-pat-saved"),
    confRootUrl: $("conf-root-url"),
    confTlsFields: $("conf-tls-fields"),
    confCaPath: $("conf-ca-path"),
    confVerifySsl: $("conf-verify-ssl"),
    confAutoSync: $("conf-auto-sync"),
    confIntervalField: $("conf-interval-field"),
    confAutoSyncInterval: $("conf-auto-sync-interval"),
    confReplaceMode: $("conf-replace-mode"),
    confSyncAttachments: $("conf-sync-attachments"),
    confSave: $("conf-save"),
    confValidate: $("conf-validate"),
    confSync: $("conf-sync"),
    confSaved: $("conf-saved"),
    confValidateResult: $("conf-validate-result"),
    confSyncCounter: $("conf-sync-counter"),
    confConsole: $("confluence-console"),
    confStatus: $("conf-status"),
    // §3.7 index maintenance (reindex / collection rebuild)
    indexCollectionState: $("index-collection-state"),
    indexSchemeWarn: $("index-scheme-warn"),
    indexBlocked: $("index-blocked"),
    indexBlockedName: $("index-blocked-name"),
    reindexBtn: $("reindex-btn"),
    reindexProgress: $("reindex-progress"),
    reindexConsole: $("reindex-console"),
    evalBlock: $("eval-block"),
    evalBtn: $("eval-btn"),
    evalDataset: $("eval-dataset"),
    evalDatasetInfo: $("eval-dataset-info"),
    evalFile: $("eval-file"),
    evalUploadBtn: $("eval-upload-btn"),
    evalLabel: $("eval-label"),
    evalLimit: $("eval-limit"),
    evalProgress: $("eval-progress"),
    evalConsole: $("eval-console"),
    evalReports: $("eval-reports"),
    rebuildBlock: $("rebuild-block"),
    rebuildUnavailable: $("rebuild-unavailable"),
    rebuildControls: $("rebuild-controls"),
    rebuildBtn: $("rebuild-btn"),
    rebuildConfirm: $("rebuild-confirm"),
    rebuildConfirmInput: $("rebuild-confirm-input"),
    rebuildCollectionName: $("rebuild-collection-name"),
    rebuildLegacyWarn: $("rebuild-legacy-warn"),
    rebuildLegacyName: $("rebuild-legacy-name"),
    rebuildGo: $("rebuild-go"),
    rebuildCancel: $("rebuild-cancel"),
    rebuildProgress: $("rebuild-progress"),
    rebuildConsole: $("rebuild-console"),
    // server-mode sections (hidden in server mode / shown read-only)
    sectionServerInfo: $("section-server-info"),
    sectionConn: $("section-conn"),
    sectionCert: $("section-cert"),
    sectionEnv: $("section-env"),
    // §M model / §S rag / §P prompts — tuning form (both modes)
    saveModel: $("save-model"),
    savedModel: $("saved-model"),
    saveRag: $("save-rag"),
    savedRag: $("saved-rag"),
    savePrompts: $("save-prompts"),
    savedPrompts: $("saved-prompts"),
    roPrompts: $("ro-prompts"),
    roPromptCondense: $("ro-prompt-condense"),
    roPromptGrader: $("ro-prompt-grader"),
    roPromptMeta: $("ro-prompt-meta"),
    roPromptMetaSelf: $("ro-prompt-meta-self"),
    // login modal (server mode)
    loginScrim: $("login-scrim"),
    loginToken: $("login-token"),
    loginToggle: $("login-toggle"),
    loginError: $("login-error"),
    loginSubmit: $("login-submit"),
    // toasts
    toasts: $("toasts"),
  };

  /* ============================ UTILS ============================ */
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function relTime(ts) {
    if (!ts) return "";
    const then = typeof ts === "number" ? ts : Date.parse(ts);
    if (isNaN(then)) return "";
    const s = Math.max(0, Math.floor((Date.now() - then) / 1000));
    if (s < 60) return "только что";
    const m = Math.floor(s / 60);
    if (m < 60) return m + " мин назад";
    const h = Math.floor(m / 60);
    if (h < 24) return h + " ч назад";
    const d = Math.floor(h / 24);
    if (d < 7) return d + " дн назад";
    try { return new Date(then).toLocaleDateString("ru-RU"); } catch (_) { return ""; }
  }
  function scrollStreamToBottom() {
    dom.stream.scrollTop = dom.stream.scrollHeight;
  }

  /* ============================ AUTH / TOKEN STORE ============================
   * Server ("multi-tenant") mode keeps a per-user CogniVault token in
   * localStorage under `cvToken`. It is attached as `Authorization: Bearer …`
   * to every /api/* call. In local mode there is no token → header omitted and
   * the backend ignores auth entirely.
   */
  const TOKEN_KEY = "cvToken";
  const THEME_KEY = "cvTheme"; // server mode stores theme client-side (no PUT /api/config)
  function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; } }
  function setToken(t) { try { localStorage.setItem(TOKEN_KEY, t); } catch (_) {} }
  function clearToken() { try { localStorage.removeItem(TOKEN_KEY); } catch (_) {} }
  function authHeaders() { const t = getToken(); return t ? { Authorization: "Bearer " + t } : {}; }
  function isServerMode() { return !!(state.config && state.config.mode === "server"); }
  function unauthorizedError() { const e = new Error("UNAUTHORIZED"); e.status = 401; e.handled = true; return e; }
  // Central 401 handler: wipe the token, hide identity, reopen the login modal.
  function handleUnauthorized(message) {
    clearToken();
    if (dom.identity) dom.identity.hidden = true;
    showLogin(message || "Токен истёк или отозван");
  }

  /* ============================ API HELPERS ============================ */
  async function apiGet(path) {
    const res = await fetch(path, { headers: Object.assign({ Accept: "application/json" }, authHeaders()) });
    if (res.status === 401 && isServerMode()) { handleUnauthorized(); throw unauthorizedError(); }
    if (!res.ok) throw new Error(path + " → " + res.status);
    return res.json();
  }
  async function apiSend(path, method, body) {
    const res = await fetch(path, {
      method,
      headers: Object.assign({ "Content-Type": "application/json", Accept: "application/json" }, authHeaders()),
      body: body == null ? undefined : JSON.stringify(body),
    });
    if (res.status === 401 && isServerMode()) { handleUnauthorized(); throw unauthorizedError(); }
    if (!res.ok) {
      // Keep the whole error envelope on the thrown error: callers that only need
      // a sentence use .message, validation-heavy ones (PUT /api/config) also
      // show .detail instead of a generic "something went wrong".
      let msg = path + " → " + res.status;
      let code = null;
      let detail = null;
      try {
        const j = await res.json();
        if (j && j.error) {
          msg = j.error.message || msg;
          code = j.error.code || null;
          detail = j.error.detail != null ? j.error.detail : null;
        }
      } catch (_) {}
      const err = new Error(msg);
      err.status = res.status;
      err.code = code;
      err.detail = detail;
      throw err;
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("application/json") ? res.json() : null;
  }

  /* ============================ SSE PARSER ============================
   * Consume a POST SSE stream via fetch + ReadableStream. onEvent({event,data}).
   * Uses streaming TextDecoder so multi-byte (Cyrillic) chars split across
   * network chunks are reassembled correctly. Blocks are separated by \n\n;
   * each block has `event:` and (possibly multi-line) `data:` fields.
   */
  async function consumeSSE(response, onEvent) {
    const reader = response.body.getReader();
    const dec = new TextDecoder("utf-8");
    let buffer = "";

    const dispatch = (block) => {
      if (!block.trim()) return;
      let ev = "message";
      const dataLines = [];
      for (const rawLine of block.split("\n")) {
        const line = rawLine.replace(/\r$/, "");
        if (!line || line.startsWith(":")) continue; // comment / blank
        if (line.startsWith("event:")) {
          ev = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^ /, ""));
        }
      }
      const dataStr = dataLines.join("\n");
      let data = dataStr;
      if (dataStr) { try { data = JSON.parse(dataStr); } catch (_) { data = dataStr; } }
      onEvent({ event: ev, data });
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        dispatch(block);
      }
    }
    // flush decoder + any trailing block
    buffer += dec.decode();
    if (buffer.indexOf("\n\n") !== -1) {
      const parts = buffer.split("\n\n");
      for (const p of parts) dispatch(p);
    } else if (buffer.trim()) {
      dispatch(buffer);
    }
  }

  /* ============================ MARKDOWN (tiny, safe) ============================
   * Supported: ``` fenced code ```, ### headings, - / 1. lists, **bold**,
   * `inline code`, [text](url) links, paragraphs. HTML is escaped FIRST.
   */
  // private-use sentinels wrap protected inline-code placeholders so the
  // digit index cannot collide with real digits in the surrounding text.
  var CODE_OPEN = "\uE000";
  var CODE_CLOSE = "\uE001";
  function renderInline(text) {
    let s = escapeHtml(text);
    // inline code first (protect its contents from other rules)
    const codes = [];
    s = s.replace(/`([^`]+)`/g, function (_m, c) {
      codes.push("<code>" + c + "</code>");
      return CODE_OPEN + (codes.length - 1) + CODE_CLOSE;
    });
    // bold
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // links [text](url) — only http(s)/relative, escaped href
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_m, t, url) {
      if (!/^(https?:\/\/|\/|mailto:)/i.test(url)) return t;
      return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + t + "</a>";
    });
    // restore inline code
    s = s.replace(/\uE000(\d+)\uE001/g, function (_m, i) { return codes[Number(i)]; });
    return s;
  }

  function renderMarkdown(text) {
    const lines = String(text).replace(/\r\n/g, "\n").split("\n");
    let html = "";
    let i = 0;
    let listType = null;     // "ul" | "ol" | null
    let paraBuf = [];

    const flushPara = () => {
      if (paraBuf.length) {
        html += "<p>" + renderInline(paraBuf.join(" ")) + "</p>";
        paraBuf = [];
      }
    };
    const closeList = () => { if (listType) { html += "</" + listType + ">"; listType = null; } };

    while (i < lines.length) {
      const line = lines[i];

      // fenced code block
      const fence = line.match(/^```(.*)$/);
      if (fence) {
        flushPara(); closeList();
        const codeLines = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) { codeLines.push(lines[i]); i++; }
        i++; // consume closing fence
        html += "<pre><code>" + escapeHtml(codeLines.join("\n")) + "</code></pre>";
        continue;
      }

      // heading ###
      const h = line.match(/^#{1,6}\s+(.*)$/);
      if (h) { flushPara(); closeList(); html += "<h3>" + renderInline(h[1]) + "</h3>"; i++; continue; }

      // unordered list
      const ul = line.match(/^\s*[-*]\s+(.*)$/);
      if (ul) {
        flushPara();
        if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; }
        html += "<li>" + renderInline(ul[1]) + "</li>";
        i++; continue;
      }
      // ordered list
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ol) {
        flushPara();
        if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; }
        html += "<li>" + renderInline(ol[1]) + "</li>";
        i++; continue;
      }

      // blank line → paragraph / list break
      if (!line.trim()) { flushPara(); closeList(); i++; continue; }

      // plain text → accumulate into paragraph
      closeList();
      paraBuf.push(line.trim());
      i++;
    }
    flushPara(); closeList();
    return html;
  }

  /* ============================ TOASTS ============================ */
  function toast(kind, title, message) {
    const t = el("div", "toast " + kind);
    const body = el("div", "tbody");
    body.appendChild(el("div", "tt", title));
    if (message) body.appendChild(el("div", "tm", message));
    t.appendChild(body);
    const x = el("button", "tx", "✕");
    x.setAttribute("aria-label", "Закрыть уведомление");
    x.addEventListener("click", () => t.remove());
    t.appendChild(x);
    dom.toasts.appendChild(t);
    if (kind !== "err") setTimeout(() => t.remove(), 4000);
  }

  /* ============================ STATUS ============================ */
  function setPill(pill, kind, label) {
    pill.className = "pill " + kind;
    const t = pill.querySelector(".pt");
    if (t) t.textContent = label;
  }

  async function refreshStatus(envSettingUp) {
    // Раздел оценки живёт своим статусом и не должен зависеть от того, дошёл ли
    // /api/status: он про харнесс в образе, а не про связь с бэкендом.
    refreshEvalSection();
    let st;
    try {
      st = await apiGet("/api/status");
    } catch (_) {
      setPill(dom.pillCogni, "err", "Нет связи");
      setPill(dom.pillCert, "err", "Нет сертификатов");
      setPill(dom.pillEnv, "warn", "Не настроено");
      updateBanner({ cognivault: { ok: false }, env: {} });
      return;
    }
    const cv = st.cognivault || {};
    const gc = st.gigachat || {};
    const env = st.env || {};
    // CogniVault
    setPill(dom.pillCogni, cv.ok ? "ok" : "err", cv.ok ? "Подключено" : "Нет связи");
    // Certificates
    const certs = gc.cert_exists && gc.key_exists;
    setPill(dom.pillCert, certs ? "ok" : "err", certs ? "Сертификаты" : "Нет сертификатов");
    // Environment
    if (envSettingUp) {
      setPill(dom.pillEnv, "warn", "Настраиваю окружение…");
    } else if (env.venv_exists) {
      setPill(dom.pillEnv, "ok", "Окружение готово");
    } else {
      setPill(dom.pillEnv, "warn", "Не настроено");
    }
    updateBanner(st);
    return st;
  }

  function updateBanner(st) {
    const cv = (st && st.cognivault) || {};
    // Server mode: env/token setup banners don't apply (admin-managed).
    // Only surface a lost-connection warning.
    if (isServerMode()) {
      if (st && cv.ok === false) {
        dom.bannerText.textContent = "Нет связи с сервером CogniVault. Попробуйте позже.";
        dom.banner.hidden = false;
      } else {
        dom.banner.hidden = true;
      }
      return;
    }
    const env = (st && st.env) || {};
    const hasToken = state.config && state.config.cognivault &&
      !!state.config.cognivault.token;
    const envReady = env.venv_exists;
    let text = "";
    if (st && cv.ok === false) {
      text = "Нет связи с сервером CogniVault. Проверьте, что служба запущена.";
    } else if (!envReady) {
      text = "Окружение не настроено. Откройте Настройки → Окружение, чтобы завершить установку.";
    } else if (!hasToken) {
      text = "Не задан токен доступа. Откройте Настройки → Подключение.";
    }
    if (text) {
      dom.bannerText.textContent = text;
      dom.banner.hidden = false;
    } else {
      dom.banner.hidden = true;
    }
  }

  /* ============================ THEME ============================ */
  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === "light" || theme === "dark") root.setAttribute("data-theme", theme);
    else root.removeAttribute("data-theme"); // auto
    dom.themeBtn.textContent = theme === "light" ? "☀" : theme === "dark" ? "☾" : "◐";
  }
  // Server mode: `ui.theme` is a per-token setting, so the server value wins once
  // we are logged in; localStorage is the pre-login / offline fallback and keeps
  // the choice from flashing on the next boot before the config lands.
  function getServerTheme() {
    // Logged in → the token's stored theme is authoritative (it follows the user
    // across browsers). Not logged in → the anonymous GET only ever carries the
    // admin default, so fall back to whatever this browser last chose.
    if (getToken()) {
      const fromServer = state.config && state.config.ui && state.config.ui.theme;
      if (fromServer) return fromServer;
    }
    try { return localStorage.getItem(THEME_KEY) || "auto"; } catch (_) { return "auto"; }
  }
  async function cycleTheme() {
    const order = ["auto", "light", "dark"];
    if (isServerMode()) {
      const cur = getServerTheme();
      const next = order[(order.indexOf(cur) + 1) % order.length];
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (_) {}
      if (state.config) {
        if (!state.config.ui) state.config.ui = {};
        state.config.ui.theme = next;
      }
      // Persist per token — but only once there IS one: a pre-login toggle would
      // 401 and bounce the user into the login dialog for a cosmetic action.
      if (getToken()) {
        try { await apiSend("/api/config", "PUT", { ui: { theme: next } }); } catch (_) {}
      }
      return;
    }
    const cur = (state.config && state.config.ui && state.config.ui.theme) || "auto";
    const next = order[(order.indexOf(cur) + 1) % order.length];
    applyTheme(next);
    if (!state.config.ui) state.config.ui = {};
    state.config.ui.theme = next;
    try {
      await apiSend("/api/config", "PUT", { ui: { theme: next } });
    } catch (e) {
      toast("err", "Не удалось сохранить тему", e.message);
    }
  }

  /* ============================ SERVER MODE (multi-tenant) ============================ */
  function showLogin(message) {
    if (dom.loginError) {
      if (message) { dom.loginError.textContent = message; dom.loginError.hidden = false; }
      else { dom.loginError.textContent = ""; dom.loginError.hidden = true; }
    }
    if (dom.loginToken) dom.loginToken.value = "";
    if (dom.loginScrim) dom.loginScrim.hidden = false;
    if (dom.loginToken) { try { dom.loginToken.focus(); } catch (_) {} }
  }
  function hideLogin() {
    if (dom.loginScrim) dom.loginScrim.hidden = true;
    if (dom.loginError) { dom.loginError.hidden = true; dom.loginError.textContent = ""; }
  }
  function showLoginError(msg) {
    if (dom.loginError) { dom.loginError.textContent = msg; dom.loginError.hidden = false; }
  }
  function toggleLoginTokenVisibility() {
    state.loginTokenVisible = !state.loginTokenVisible;
    dom.loginToken.type = state.loginTokenVisible ? "text" : "password";
    dom.loginToggle.textContent = state.loginTokenVisible ? "Скрыть" : "Показать";
    dom.loginToggle.setAttribute("aria-pressed", String(state.loginTokenVisible));
  }

  // GET /api/whoami — 200 {userId,ok}; 401 invalid; 503 CV unavailable.
  async function whoami(token) {
    const t = token || getToken();
    const res = await fetch("/api/whoami", {
      headers: { Accept: "application/json", Authorization: "Bearer " + t },
    });
    if (res.ok) return res.json();
    const err = new Error("whoami → " + res.status);
    err.status = res.status;
    throw err;
  }

  async function submitLogin() {
    const token = (dom.loginToken.value || "").trim();
    if (!token) { showLoginError("Введите токен"); return; }
    dom.loginSubmit.disabled = true;
    try {
      const who = await whoami(token);
      setToken(token);
      hideLogin();
      onLoggedIn(who);
    } catch (e) {
      if (e.status === 401) showLoginError("Неверный токен");
      else if (e.status === 503) showLoginError("CogniVault недоступен, попробуйте позже");
      else showLoginError("Не удалось войти. Попробуйте ещё раз.");
    } finally {
      dom.loginSubmit.disabled = false;
    }
  }

  // Boot path for server mode: verify an existing token or prompt for login.
  async function bootServer() {
    const token = getToken();
    if (!token) { showLogin(); return; }
    try {
      const who = await whoami(token);
      onLoggedIn(who);
    } catch (e) {
      if (e.status === 503) {
        showLogin("CogniVault недоступен, попробуйте позже");
      } else {
        clearToken();
        showLogin(e.status === 401 ? "Токен истёк или отозван" : undefined);
      }
    }
  }

  async function onLoggedIn(who) {
    const userId = (who && (who.userId || who.user_id)) || "";
    if (dom.identityUser) dom.identityUser.textContent = userId;
    if (dom.identity) dom.identity.hidden = false;
    // Re-read the config with the user's token: the anonymous boot request can
    // legitimately carry a narrower subset than the authenticated one.
    try { state.config = await apiGet("/api/config"); } catch (_) {}
    applyServerConfig();
    refreshStatus();
    loadHistory();
  }

  function logout() {
    clearToken();
    if (dom.identity) dom.identity.hidden = true;
    // reset the visible conversation so the next user starts clean
    if (state.abort) state.abort.abort();
    state.chatId = null;
    state.messages = [];
    renderWelcome();
    if (dom.chatlist) dom.chatlist.textContent = "";
    if (dom.historyCount) dom.historyCount.textContent = "";
    showLogin();
  }

  // Hide admin-managed / local-only UI; reveal the read-only info section.
  function applyServerModeUI() {
    if (dom.sectionConn) dom.sectionConn.hidden = true;
    if (dom.sectionCert) dom.sectionCert.hidden = true;
    if (dom.sectionEnv) dom.sectionEnv.hidden = true;
    if (dom.sectionServerInfo) dom.sectionServerInfo.hidden = false;
    // cert/env pills are admin/local concerns — hide them
    if (dom.pillCert) dom.pillCert.hidden = true;
    if (dom.pillEnv) dom.pillEnv.hidden = true;
  }

  // Apply the safe config subset returned in server mode. Model / RAG / prompt
  // knobs are editable here too — only the fields listed in `locked` stay
  // read-only, so the drawer shows a real form instead of static text.
  function applyServerConfig() {
    const c = state.config || {};
    const rag = c.rag || {};
    state.rag = rag.default_on !== false;
    applyRagUI();
    applyTheme(getServerTheme());
    bindTuningForm();
  }

  /* ============================ SETTINGS ============================ */
  function bindConfigToForm() {
    const c = state.config || {};
    const cv = c.cognivault || {};
    const giga = c.gigachat || {};
    dom.cfgCognivaultUrl.value = cv.base_url || "";
    dom.cfgToken.value = cv.token || "";
    dom.cfgGigaUrl.value = giga.base_url || "";
    dom.cfgModel.value = giga.model || "GigaChat-3-Ultra-preview";
    dom.cfgCert.value = giga.cert_path || "";
    dom.cfgKey.value = giga.key_path || "";
    dom.cfgPassphrase.value = giga.key_passphrase || "";
    dom.cfgCa.value = giga.ca_path || "";
    dom.cfgVerifySsl.checked = giga.verify_ssl !== false;

    // env / SberOSC mirror
    const env = c.env || {};
    if (dom.cfgPipIndex) dom.cfgPipIndex.value = env.pip_index_url || "";
    if (dom.cfgPipToken) dom.cfgPipToken.value = env.pip_token || "";

    // RAG defaults
    const rag = c.rag || {};
    state.rag = rag.default_on !== false;
    applyRagUI();

    // model / RAG / prompt tuning form
    bindTuningForm();

    // theme
    applyTheme((c.ui && c.ui.theme) || "auto");
  }

  function flashSaved(node) {
    node.hidden = false;
    setTimeout(() => { node.hidden = true; }, 2500);
  }

  async function saveConnection() {
    const payload = {
      cognivault: {
        base_url: dom.cfgCognivaultUrl.value.trim(),
        token: dom.cfgToken.value,
      },
    };
    dom.saveConn.disabled = true;
    try {
      const updated = await apiSend("/api/config", "PUT", payload);
      if (updated) state.config = updated; else Object.assign(state.config.cognivault || (state.config.cognivault = {}), payload.cognivault);
      flashSaved(dom.savedConn);
      toast("ok", "Настройки сохранены", "Подключение обновлено");
      await refreshStatus();
    } catch (e) {
      toast("err", "Ошибка сохранения", e.message);
    } finally {
      dom.saveConn.disabled = false;
    }
  }

  async function saveCertificates() {
    const payload = {
      gigachat: {
        base_url: dom.cfgGigaUrl.value.trim(),
        model: dom.cfgModel.value.trim(),
        cert_path: dom.cfgCert.value.trim(),
        key_path: dom.cfgKey.value.trim(),
        key_passphrase: dom.cfgPassphrase.value,
        ca_path: dom.cfgCa.value.trim(),
        verify_ssl: dom.cfgVerifySsl.checked,
      },
    };
    dom.saveCert.disabled = true;
    try {
      const updated = await apiSend("/api/config", "PUT", payload);
      if (updated) state.config = updated; else state.config.gigachat = payload.gigachat;
      bindTuningForm(); // §2 writes gigachat.model too — keep §M in sync
      flashSaved(dom.savedCert);
      toast("ok", "Настройки сохранены", "Сертификаты обновлены");
      await refreshStatus();
    } catch (e) {
      toast("err", "Ошибка сохранения", e.message);
    } finally {
      dom.saveCert.disabled = false;
    }
  }

  function toggleTokenVisibility() {
    state.tokenVisible = !state.tokenVisible;
    dom.cfgToken.type = state.tokenVisible ? "text" : "password";
    dom.tokenToggle.textContent = state.tokenVisible ? "Скрыть" : "Показать";
    dom.tokenToggle.setAttribute("aria-pressed", String(state.tokenVisible));
  }

  function togglePipTokenVisibility() {
    state.pipTokenVisible = !state.pipTokenVisible;
    dom.cfgPipToken.type = state.pipTokenVisible ? "text" : "password";
    dom.pipTokenToggle.textContent = state.pipTokenVisible ? "Скрыть" : "Показать";
    dom.pipTokenToggle.setAttribute("aria-pressed", String(state.pipTokenVisible));
  }

  async function saveEnvMirror() {
    const payload = {
      env: {
        pip_index_url: dom.cfgPipIndex.value.trim(),
        pip_token: dom.cfgPipToken.value,
      },
    };
    dom.saveEnvMirror.disabled = true;
    try {
      const updated = await apiSend("/api/config", "PUT", payload);
      if (updated) state.config = updated;
      else Object.assign(state.config.env || (state.config.env = {}), payload.env);
      flashSaved(dom.savedEnvMirror);
      toast("ok", "Настройки сохранены", "Зеркало PyPI обновлено");
    } catch (e) {
      toast("err", "Ошибка сохранения", e.message);
    } finally {
      dom.saveEnvMirror.disabled = false;
    }
  }

  /* ============================ MODEL / RAG / PROMPT TUNING ============================
   * Three drawer sections (§M model, §S search, §P prompts) driven by descriptor
   * tables instead of hand-written per-field code. A descriptor is
   *   { key, id, type, label, def?, locked? }
   * where `type` decides coercion on the way out: "text" stays a string,
   * "int"/"number" become numbers, "bool" a boolean. Each section saves on its
   * own button and PUTs exactly one subtree of /api/config.
   */
  // One provider, one model — the model key follows the provider, because the
  // two transports keep their names apart (switching back must not have
  // clobbered the other one). The user never sees that split: they pick a
  // provider and a model, and the form writes whichever key is live.
  function modelKeyFor(provider) {
    return provider === "kitai" ? "kitai_model" : "model";
  }

  const MODEL_FIELDS = [
    { key: "provider", id: "cfg-gc-provider", type: "text", label: "Провайдер" },
    { key: "model", id: "cfg-gc-model", type: "text", label: "Модель" },
    { key: "temperature", id: "cfg-gc-temperature", type: "number", label: "Температура" },
    { key: "max_tokens", id: "cfg-gc-max-tokens", type: "int", label: "Максимум токенов в ответе" },
    // the context window is sized by the deployment, never by the user
    { key: "model_context_tokens", id: "cfg-gc-context-tokens", type: "int", label: "Окно контекста модели", locked: true },
  ];

  const RAG_FIELDS = [
    { key: "default_on", id: "cfg-rag-default-on", type: "bool", label: "Поиск по умолчанию", def: true },
    { key: "limit", id: "cfg-rag-limit", type: "int", label: "Фрагментов в ответе" },
    { key: "rerank_candidates", id: "cfg-rag-rerank-candidates", type: "int", label: "Кандидатов на отбор" },
    { key: "grader_enabled", id: "cfg-rag-grader-enabled", type: "bool", label: "Оценка релевантности" },
    { key: "grader_threshold", id: "cfg-rag-grader-threshold", type: "int", label: "Порог оценки" },
    { key: "grader_keep_top", id: "cfg-rag-grader-keep-top", type: "int", label: "Оставлять лучших" },
    { key: "condense_enabled", id: "cfg-rag-condense-enabled", type: "bool", label: "Уточнение вопроса" },
    { key: "condense_first_turn", id: "cfg-rag-condense-first-turn", type: "bool", label: "Разбор первого вопроса" },
    { key: "corpus_tree_enabled", id: "cfg-rag-corpus-tree-enabled", type: "bool", label: "Дерево разделов в контексте" },
    { key: "max_context_chars", id: "cfg-rag-max-context-chars", type: "int", label: "Бюджет контекста" },
    { key: "file_full_chars", id: "cfg-rag-file-full-chars", type: "int", label: "Файл целиком до" },
    { key: "section_max_chars", id: "cfg-rag-section-max-chars", type: "int", label: "Раздел не длиннее" },
    { key: "max_expanded_files", id: "cfg-rag-max-expanded-files", type: "int", label: "Файлов раскрывать" },
    { key: "min_score", id: "cfg-rag-min-score", type: "number", label: "Минимальная близость" },
  ];

  const PROMPT_FIELDS = [
    { key: "system", id: "cfg-prompt-system", resetId: "prompt-system-reset", dirtyId: "prompt-system-dirty" },
    { key: "context_reminder", id: "cfg-prompt-reminder", resetId: "prompt-reminder-reset", dirtyId: "prompt-reminder-dirty" },
  ];

  // Descriptors cache their node (null included) after the first lookup.
  function fieldNode(f) {
    if (f.node === undefined) f.node = $(f.id);
    return f.node;
  }
  function promptFlagNode(f) {
    if (f.flag === undefined) f.flag = $(f.dirtyId);
    return f.flag;
  }

  // A field is read-only when the descriptor pins it or the server lists its
  // dotted path in `locked`. The array may be absent — then nothing is locked.
  function fieldLocked(f, prefix) {
    if (f.locked) return true;
    const locked = state.config?.locked;
    return Array.isArray(locked) && locked.indexOf(prefix + "." + f.key) !== -1;
  }

  // Disable a locked input and keep exactly one "set by the admin" note next to
  // it. Disabled inputs also drop out of the drawer focus trap, which is right.
  function applyFieldLock(f, prefix) {
    const node = fieldNode(f);
    if (!node || !node.parentNode) return;
    const on = fieldLocked(f, prefix);
    node.disabled = on;
    const holder = node.parentNode;
    const note = holder.querySelector(".lock-hint");
    if (on && !note) holder.appendChild(el("small", "hint lock-hint", "Значение задаёт администратор"));
    else if (!on && note) note.remove();
  }

  function bindFields(fields, prefix, values) {
    const v = values || {};
    fields.forEach((f) => {
      const node = fieldNode(f);
      if (!node) return;
      const raw = v[f.key];
      if (f.type === "bool") node.checked = raw == null ? !!f.def : !!raw;
      else node.value = raw == null ? "" : String(raw);
      applyFieldLock(f, prefix);
    });
  }

  // Build one PUT subtree out of a descriptor table. Numbers are coerced here —
  // the backend rejects a string with 400 rather than repairing it. Empty inputs
  // are omitted entirely so an untouched field never clears a stored value.
  // Returns the labels of fields whose text is not a valid number as `invalid`.
  function collectFields(fields, prefix) {
    const payload = {};
    const invalid = [];
    fields.forEach((f) => {
      const node = fieldNode(f);
      if (!node || fieldLocked(f, prefix)) return; // never echo admin-managed values back
      if (f.type === "bool") { payload[f.key] = !!node.checked; return; }
      const raw = String(node.value == null ? "" : node.value).trim();
      if (!raw) return;
      if (f.type === "text") { payload[f.key] = raw; return; }
      const num = Number(raw.replace(",", "."));
      if (!Number.isFinite(num) || (f.type === "int" && !Number.isInteger(num))) {
        invalid.push(f.label);
        return;
      }
      payload[f.key] = num;
    });
    return { payload: payload, invalid: invalid };
  }

  function promptDefault(key) {
    const d = state.config?.defaults?.prompts;
    return d && typeof d[key] === "string" ? d[key] : null;
  }
  // "изменён" badge — only meaningful once the server told us the built-in text.
  function refreshPromptDirty(f) {
    const node = fieldNode(f);
    const flag = promptFlagNode(f);
    if (!node || !flag) return;
    const def = promptDefault(f.key);
    flag.hidden = def == null || node.value === def;
  }

  function bindPrompts() {
    const prompts = state.config?.prompts || {};
    PROMPT_FIELDS.forEach((f) => {
      const node = fieldNode(f);
      if (!node) return;
      node.value = typeof prompts[f.key] === "string" ? prompts[f.key] : "";
      refreshPromptDirty(f);
    });
    // Service prompts are read-only — condense/grader because their replies are
    // parsed as JSON, meta/meta_self because they are what keeps an ungrounded
    // answer from being generated. Shown only when the server exposes them: an
    // older backend sends condense/grader alone.
    const ro = state.config?.readonly?.prompts;
    const hasRo = !!(ro && (ro.condense || ro.grader || ro.meta || ro.meta_self));
    if (dom.roPrompts) dom.roPrompts.hidden = !hasRo;
    if (!hasRo) return;
    if (dom.roPromptCondense) dom.roPromptCondense.value = ro.condense || "";
    if (dom.roPromptGrader) dom.roPromptGrader.value = ro.grader || "";
    if (dom.roPromptMeta) dom.roPromptMeta.value = ro.meta || "";
    if (dom.roPromptMetaSelf) dom.roPromptMetaSelf.value = ro.meta_self || "";
  }

  function bindTuningForm() {
    const c = state.config || {};
    const provider = (c.gigachat || {}).provider || "gigachat";
    MODEL_FIELDS[1].key = modelKeyFor(provider);
    bindFields(MODEL_FIELDS, "gigachat", c.gigachat);
    bindProviderChange();
    refreshModelChoices();
    bindFields(RAG_FIELDS, "rag", c.rag);
    bindPrompts();
  }

  // Adopt a GET/PUT /api/config response and surface everything the server said
  // about it. `warnings` / `ignored` may be absent — both are optional.
  function applyConfigResponse(updated) {
    if (!updated || typeof updated !== "object") return;
    state.config = updated;
    const warnings = Array.isArray(updated.warnings) ? updated.warnings : [];
    warnings.forEach((w) => toast("warn", "Внимание", String(w)));
    const ignored = Array.isArray(updated.ignored) ? updated.ignored : [];
    if (ignored.length) toast("warn", "Часть настроек не применена", ignored.join(", "));
    bindTuningForm();
  }

  // 400 from PUT /api/config carries {code,message,detail} — show the detail too.
  function configErrorText(e) {
    const detail = e && e.detail;
    let extra = "";
    if (typeof detail === "string") extra = detail;
    else if (detail != null) { try { extra = JSON.stringify(detail); } catch (_) { extra = String(detail); } }
    return [(e && e.message) || "Не удалось сохранить", extra].filter(Boolean).join(" — ");
  }

  async function putConfigSubtree(body, btn, savedNode, okMessage) {
    if (btn) btn.disabled = true;
    try {
      const updated = await apiSend("/api/config", "PUT", body);
      applyConfigResponse(updated);
      if (savedNode) flashSaved(savedNode);
      toast("ok", "Настройки сохранены", okMessage);
    } catch (e) {
      if (!e.handled) toast("err", "Ошибка сохранения", configErrorText(e));
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function saveFieldSection(fields, prefix, btn, savedNode, okMessage) {
    const collected = collectFields(fields, prefix);
    if (collected.invalid.length) {
      toast("warn", "Проверьте значения", "Ожидается число: " + collected.invalid.join(", "));
      return;
    }
    const body = {};
    body[prefix] = collected.payload;
    await putConfigSubtree(body, btn, savedNode, okMessage);
  }

  // Switching the provider re-points the model field at that provider's stored
  // name and re-fetches its catalogue. Done live, not on save: otherwise the
  // user picks KitAI, still sees the GigaChat model name, and reasonably
  // concludes the switch did nothing.
  function bindProviderChange() {
    const sel = document.getElementById("cfg-gc-provider");
    if (!sel || sel.dataset.bound) return;
    sel.dataset.bound = "1";
    sel.addEventListener("change", () => {
      const key = modelKeyFor(sel.value);
      MODEL_FIELDS[1].key = key;
      const stored = (state.config && state.config.gigachat) || {};
      const input = document.getElementById("cfg-gc-model");
      if (input) input.value = stored[key] == null ? "" : String(stored[key]);
      refreshModelChoices();
    });
  }

  // Turn the model field into a dropdown when the provider publishes a
  // catalogue, and leave it as free text when it does not.
  //
  // Both providers publish a catalogue (KitAI `/api/v1/meta/model`, GigaChat the
  // OpenAI-compatible `{base_url}/models`), so a missing list means the call
  // failed — not that the provider offers nothing. Either way the field falls
  // back to free text: an empty dropdown with no way to type a name would lock
  // the user out of a setting because a listing call timed out.
  async function refreshModelChoices() {
    const input = document.getElementById("cfg-gc-model");
    const select = document.getElementById("cfg-gc-model-select");
    const hint = document.getElementById("cfg-gc-model-hint");
    if (!input || !select) return;

    const asText = (note) => {
      select.hidden = true;
      input.hidden = false;
      if (hint) hint.textContent = note;
    };
    asText("Загружаю список моделей…");

    let payload;
    try {
      payload = await apiGet("/api/config/models");
    } catch (e) {
      asText("Список моделей недоступен — впишите имя вручную");
      return;
    }
    const models = payload && payload.models;
    if (!Array.isArray(models) || !models.length) {
      asText(
        payload && payload.error
          ? payload.error + " — впишите имя модели вручную"
          : "Модель, которая отвечает в чате"
      );
      return;
    }

    const current = input.value.trim();
    select.innerHTML = "";
    // The stored value may not be in the catalogue (renamed, retired, or typed
    // by hand). Dropping it would silently rewrite the setting on the next save.
    if (current && !models.some((m) => m.name === current)) {
      select.appendChild(new Option(current + " — сейчас выбрана, нет в списке", current));
    }
    models.forEach((m) => select.appendChild(new Option(m.label, m.name)));
    select.value = current || models[0].name;
    input.value = select.value;
    select.onchange = () => { input.value = select.value; };
    select.hidden = false;
    input.hidden = true;
    if (hint) hint.textContent = "Список получен от провайдера";
  }

  function saveModelSettings() {
    return saveFieldSection(MODEL_FIELDS, "gigachat", dom.saveModel, dom.savedModel, "Параметры модели обновлены");
  }
  function saveRagSettings() {
    return saveFieldSection(RAG_FIELDS, "rag", dom.saveRag, dom.savedRag, "Параметры поиска обновлены");
  }

  // An empty textarea is not a reset — resetting is the button below it, which
  // sends null explicitly.
  function savePromptSettings() {
    const prompts = {};
    PROMPT_FIELDS.forEach((f) => {
      const node = fieldNode(f);
      if (!node) return;
      const v = node.value;
      if (v.trim()) prompts[f.key] = v;
    });
    return putConfigSubtree({ prompts: prompts }, dom.savePrompts, dom.savedPrompts, "Промпты обновлены");
  }
  function resetPrompt(f) {
    const prompts = {};
    prompts[f.key] = null; // null = back to the built-in default
    return putConfigSubtree({ prompts: prompts }, dom.savePrompts, dom.savedPrompts, "Промпт сброшен к стандартному");
  }

  /* ---- drawer open/close + focus trap ---- */
  let lastFocused = null;
  function openDrawer() {
    lastFocused = document.activeElement;
    dom.scrim.hidden = false;
    dom.drawer.hidden = false;
    document.addEventListener("keydown", drawerKeydown, true);
    // Load the Confluence source config each time the drawer opens (both modes).
    loadConfluenceConfig();
    // …and the index state, reattaching to a reindex/rebuild already running.
    loadIndexState();
    const first = dom.drawer.querySelector("button, input, select, a[href], textarea");
    if (first) first.focus();
  }
  function closeDrawer() {
    dom.scrim.hidden = true;
    dom.drawer.hidden = true;
    document.removeEventListener("keydown", drawerKeydown, true);
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }
  function drawerKeydown(e) {
    if (e.key === "Escape") { e.preventDefault(); closeDrawer(); return; }
    if (e.key === "Tab") {
      const items = dom.drawer.querySelectorAll(
        'button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])'
      );
      const list = Array.prototype.filter.call(items, (n) => !n.disabled && n.offsetParent !== null);
      if (!list.length) return;
      const first = list[0], last = list[list.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  }

  /* ============================ UPLOAD ============================ */
  function handleUploadFile(file) {
    if (!file) return;
    if (!/\.zip$/i.test(file.name)) {
      toast("err", "Неверный формат", "Загрузите ZIP-архив (.zip)");
      return;
    }
    uploadZip(file);
  }
  async function uploadZip(file) {
    dom.uploadResult.hidden = false;
    dom.uploadResult.textContent = "";
    dom.uploadResult.appendChild(el("span", "pill warn", "Загрузка…"));
    dom.uploadFilelist.hidden = true;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd, headers: authHeaders() });
      if (res.status === 401 && isServerMode()) {
        dom.uploadResult.textContent = "";
        handleUnauthorized();
        return;
      }
      if (!res.ok) {
        let msg = "HTTP " + res.status;
        try { const j = await res.json(); if (j && j.error) msg = j.error.message || msg; } catch (_) {}
        throw new Error(msg);
      }
      const r = await res.json();
      renderUploadResult(r);
      toast("ok", "Файлы загружены", "Архив обработан");
    } catch (e) {
      dom.uploadResult.textContent = "";
      dom.uploadResult.appendChild(el("span", "pill err", "Ошибка загрузки"));
      toast("err", "Ошибка загрузки", e.message);
    }
  }
  function renderUploadResult(r) {
    r = r || {};
    const files = r.files || [];
    const uploaded = r.uploaded != null ? r.uploaded : files.length;
    const skipped = r.skipped != null ? r.skipped : 0;
    const total = files.length;
    dom.uploadResult.hidden = false;
    dom.uploadResult.textContent = "";
    const p = el("span", "pill ok");
    p.appendChild(el("span", "dot"));
    p.appendChild(document.createTextNode(
      "Загружено: " + uploaded + " · Пропущено: " + skipped + " · Файлов: " + total));
    dom.uploadResult.appendChild(p);
    if (files.length) {
      dom.uploadFiles.textContent = "";
      files.forEach((f) => dom.uploadFiles.appendChild(el("li", null, typeof f === "string" ? f : (f.path || f.name || ""))));
      dom.uploadFilelist.hidden = false;
    }
  }

  /* ============================ ENVIRONMENT ============================ */
  function logConsole(kind, text, isStep) {
    const line = el("span", isStep ? "step" : "ln");
    if (isStep) {
      line.textContent = "▸ " + text;
    } else {
      // colorize INFO/WARN/ERR prefix if present
      const m = String(text).match(/^\s*(INFO|WARN(?:ING)?|ERR(?:OR)?)\b[:\s]?/i);
      if (m) {
        const lv = m[1].toUpperCase();
        const cls = lv.startsWith("ERR") ? "lv-err" : lv.startsWith("WARN") ? "lv-warn" : "lv-ok";
        const tag = el("span", cls, m[0]);
        line.appendChild(tag);
        line.appendChild(document.createTextNode(text.slice(m[0].length)));
      } else if (kind === "err") {
        line.className = "ln lv-err"; line.textContent = text;
      } else {
        line.textContent = text;
      }
    }
    dom.console.appendChild(line);
    dom.console.scrollTop = dom.console.scrollHeight;
  }

  async function runEnvSetup() {
    dom.envSetupBtn.disabled = true;
    dom.console.textContent = "";
    logConsole("info", "Запуск настройки окружения…", true);
    refreshStatus(true); // show "Настраиваю окружение…"
    try {
      const res = await fetch("/api/env/setup", { method: "POST", headers: Object.assign({ Accept: "text/event-stream" }, authHeaders()) });
      if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
      await consumeSSE(res, (ev) => {
        const d = ev.data || {};
        if (ev.event === "step") {
          logConsole("info", (d.label || d.name || String(d)), true);
        } else if (ev.event === "log") {
          logConsole(d.level === "error" ? "err" : "info", d.line != null ? d.line : (d.message != null ? d.message : String(d)));
        } else if (ev.event === "done") {
          logConsole("ok", "Готово.", true);
          toast("ok", "Окружение настроено", "Установка завершена");
        } else if (ev.event === "error") {
          logConsole("err", (d.message || d.code || "Ошибка"));
          toast("err", "Ошибка настройки", d.message || d.code || "");
        }
      });
    } catch (e) {
      logConsole("err", "Ошибка: " + e.message);
      toast("err", "Ошибка настройки окружения", e.message);
    } finally {
      dom.envSetupBtn.disabled = false;
      await refreshStatus();
    }
  }

  async function importEnv() {
    const path = dom.envImportPath.value.trim();
    if (!path) { toast("warn", "Укажите путь", "Введите путь к экспортированному окружению"); return; }
    dom.envImportBtn.disabled = true;
    try {
      const r = await apiSend("/api/env/import", "POST", { path });
      dom.envImportResult.hidden = false;
      dom.envImportResult.textContent = "";
      const imported = (r && r.imported) || [];
      dom.envImportResult.appendChild(el("div", null, "Импортировано: " + imported.length));
      if (imported.length) {
        const ul = el("ul");
        imported.forEach((f) => ul.appendChild(el("li", null, String(f))));
        dom.envImportResult.appendChild(ul);
      }
      if (r && r.backup) {
        dom.envImportResult.appendChild(el("div", "backup", "Резервная копия: " + r.backup));
      }
      toast("ok", "Окружение импортировано", path);
      await refreshStatus();
    } catch (e) {
      toast("err", "Ошибка импорта", e.message);
    } finally {
      dom.envImportBtn.disabled = false;
    }
  }

  /* ============================ CONFLUENCE SOURCE ============================ */
  function confServerLocked() {
    return (state.confConfig && state.confConfig.mode === "server") || isServerMode();
  }

  function applyConfAuthMode(mode) {
    state.confAuthMode = mode === "pat" ? "pat" : "basic";
    const basic = state.confAuthMode === "basic";
    dom.confAuthBasic.setAttribute("aria-pressed", String(basic));
    dom.confAuthPat.setAttribute("aria-pressed", String(!basic));
    dom.confBasicFields.hidden = !basic;
    dom.confPatFields.hidden = basic;
  }

  function applyConfInterval() {
    const on = dom.confAutoSync.checked;
    dom.confIntervalField.style.opacity = on ? "1" : ".5";
    dom.confAutoSyncInterval.disabled = !on;
  }

  function toggleConfPasswordVisibility() {
    state.confPasswordVisible = !state.confPasswordVisible;
    dom.confPassword.type = state.confPasswordVisible ? "text" : "password";
    dom.confPasswordToggle.textContent = state.confPasswordVisible ? "Скрыть" : "Показать";
    dom.confPasswordToggle.setAttribute("aria-pressed", String(state.confPasswordVisible));
  }
  function toggleConfPatVisibility() {
    state.confPatVisible = !state.confPatVisible;
    dom.confPat.type = state.confPatVisible ? "text" : "password";
    dom.confPatToggle.textContent = state.confPatVisible ? "Скрыть" : "Показать";
    dom.confPatToggle.setAttribute("aria-pressed", String(state.confPatVisible));
  }

  async function loadConfluenceConfig() {
    let c;
    try { c = await apiGet("/api/confluence/config"); } catch (_) { return; }
    state.confConfig = c || {};
    c = state.confConfig;
    const server = c.mode === "server";
    dom.confLogin.value = c.login || "";
    dom.confRootUrl.value = c.root_url || "";
    dom.confCaPath.value = c.ca_path || "";
    dom.confVerifySsl.checked = c.verify_ssl !== false;
    dom.confAutoSync.checked = !!c.auto_sync;
    dom.confAutoSyncInterval.value = c.auto_sync_interval_min != null ? c.auto_sync_interval_min : "";
    dom.confReplaceMode.checked = !!c.replace_mode;
    dom.confSyncAttachments.checked = !!c.sync_attachments;
    // secrets are never returned — clear the inputs, surface a "saved" hint
    dom.confPassword.value = "";
    dom.confPat.value = "";
    dom.confPasswordSaved.hidden = !c.has_password;
    dom.confPatSaved.hidden = !c.has_pat;
    applyConfAuthMode(c.auth_mode === "pat" ? "pat" : "basic");
    // admin-locked fields (server mode): CA/TLS hidden. The REST base is derived
    // from the root link, so there is no base-url field to lock.
    dom.confTlsFields.hidden = server;
    applyConfInterval();
    dom.confValidateResult.hidden = true;
    dom.confValidateResult.textContent = "";
    await loadConfluenceStatus();
  }

  async function saveConfluenceConfig() {
    const server = confServerLocked();
    const payload = {
      auth_mode: state.confAuthMode,
      login: dom.confLogin.value.trim(),
      root_url: dom.confRootUrl.value.trim(),
      auto_sync: dom.confAutoSync.checked,
      replace_mode: dom.confReplaceMode.checked,
      sync_attachments: dom.confSyncAttachments.checked,
    };
    const iv = parseInt(dom.confAutoSyncInterval.value, 10);
    if (!isNaN(iv)) payload.auto_sync_interval_min = iv;
    if (!server) {
      // ca_path / verify_ssl are admin-locked in server mode → omit. base_url is
      // never sent: the backend derives it from root_url automatically.
      payload.ca_path = dom.confCaPath.value.trim();
      payload.verify_ssl = dom.confVerifySsl.checked;
    }
    // never clear a saved secret — only send when the user typed something
    const pw = dom.confPassword.value;
    const pat = dom.confPat.value;
    if (pw) payload.password = pw;
    if (pat) payload.pat = pat;
    dom.confSave.disabled = true;
    try {
      await apiSend("/api/confluence/config", "PUT", payload);
      flashSaved(dom.confSaved);
      toast("ok", "Настройки сохранены", "Источник Confluence обновлён");
      await loadConfluenceConfig();
    } catch (e) {
      toast("err", "Ошибка сохранения", e.message);
    } finally {
      dom.confSave.disabled = false;
    }
  }

  const CONF_VALIDATE_ERR = {
    BAD_URL: "Неверный адрес или ссылка на страницу",
    AUTH_FAILED_BASIC_SSO: "Basic отключён — переключитесь на токен (PAT)",
    PAGE_NOT_FOUND: "Корневая страница не найдена",
    TLS_ERROR: "Ошибка TLS-сертификата сервера",
    CONF_UNAVAILABLE: "Confluence недоступен, попробуйте позже",
  };

  function showConfValidate(kind, text) {
    dom.confValidateResult.hidden = false;
    dom.confValidateResult.textContent = "";
    const p = el("span", "pill " + kind);
    p.appendChild(el("span", "dot"));
    p.appendChild(document.createTextNode(text));
    dom.confValidateResult.appendChild(p);
  }

  async function validateConfluence() {
    dom.confValidate.disabled = true;
    showConfValidate("warn", "Проверка подключения…");
    try {
      const res = await fetch("/api/confluence/validate", {
        method: "POST",
        headers: Object.assign({ "Content-Type": "application/json", Accept: "application/json" }, authHeaders()),
        body: JSON.stringify({}),
      });
      if (res.status === 401 && isServerMode()) { handleUnauthorized(); throw unauthorizedError(); }
      let body = null;
      try { body = await res.json(); } catch (_) {}
      if (res.ok && body && body.ok) {
        const est = body.page_count_estimate != null ? body.page_count_estimate : "?";
        const mode = body.auth_mode_used || state.confAuthMode;
        const title = body.root_title || "Корневая страница";
        const space = body.space ? " · " + body.space : "";
        showConfValidate("ok", "✓ " + title + space + ", ~" + est + " страниц (вход: " + mode + ")");
      } else {
        const err = (body && body.error) || {};
        showConfValidate("err", CONF_VALIDATE_ERR[err.code] || err.message || "Не удалось проверить подключение");
      }
    } catch (e) {
      if (!e.handled) showConfValidate("err", "Ошибка соединения: " + e.message);
    } finally {
      dom.confValidate.disabled = false;
    }
  }

  async function loadConfluenceStatus() {
    let st;
    try { st = await apiGet("/api/confluence/status"); } catch (_) { return; }
    st = st || {};
    const parts = [];
    if (st.last_sync_at) {
      const when = relTime(st.last_sync_at) || String(st.last_sync_at);
      parts.push("Последняя синхронизация: " + when);
    } else {
      parts.push("Последняя синхронизация: ещё не выполнялась");
    }
    if (st.page_count != null) {
      const noun = pluralRu(st.page_count, "страница", "страницы", "страниц");
      parts.push(st.page_count + " " + noun);
    }
    if (st.root_title) parts.push(st.root_title);
    dom.confStatus.textContent = parts.join(" · ");
  }

  // Console rendering for the confluence sync SSE (separate from env #console).
  function confLog(kind, text) {
    const cls = kind === "error" ? "ln lv-err"
      : kind === "done" ? "step lv-ok"
      : kind === "step" ? "step"
      : "ln lv-dim"; // log
    const line = el("span", cls);
    const prefix = (kind === "step" || kind === "done") ? "▸ " : "";
    line.textContent = prefix + text;
    dom.confConsole.appendChild(line);
    dom.confConsole.scrollTop = dom.confConsole.scrollHeight;
  }

  const CONF_PAGE_ACTION = {
    new: { label: "создано", cls: "lv-ok" },
    updated: { label: "обновлено", cls: "lv-teal" },
    skipped: { label: "пропущено", cls: "lv-dim" },
    deleted: { label: "удалено", cls: "lv-warn" },
    failed: { label: "ошибка", cls: "lv-err" },
  };
  function renderConfPage(d) {
    d = d || {};
    const a = CONF_PAGE_ACTION[d.action] || { label: String(d.action || "?"), cls: "lv-dim" };
    const line = el("span", "ln");
    line.appendChild(el("span", a.cls, "+ " + a.label + ": "));
    const total = d.total != null ? d.total : "?";
    const index = d.index != null ? d.index : "?";
    line.appendChild(document.createTextNode((d.title || "Без названия") + " (" + index + "/" + total + ")"));
    dom.confConsole.appendChild(line);
    dom.confConsole.scrollTop = dom.confConsole.scrollHeight;
    if (d.total != null) {
      dom.confSyncCounter.hidden = false;
      dom.confSyncCounter.textContent = "Обработано " + index + " из " + total;
    }
  }

  async function syncConfluence() {
    if (state.confSyncing) return;
    const replace = dom.confReplaceMode.checked;
    state.confSyncing = true;
    dom.confSync.disabled = true;
    dom.confConsole.hidden = false;
    dom.confConsole.textContent = "";
    dom.confSyncCounter.hidden = false;
    dom.confSyncCounter.textContent = "";
    confLog("step", "Запуск синхронизации…");
    try {
      const res = await fetch("/api/confluence/sync", {
        method: "POST",
        headers: Object.assign({ "Content-Type": "application/json", Accept: "text/event-stream" }, authHeaders()),
        body: JSON.stringify({ replace: replace }),
      });
      if (res.status === 401 && isServerMode()) { handleUnauthorized(); return; }
      if (!res.ok || !res.body) {
        let msg = "HTTP " + res.status;
        try { const j = await res.json(); if (j && j.error) msg = j.error.message || j.error.code || msg; } catch (_) {}
        throw new Error(msg);
      }
      await consumeSSE(res, (ev) => {
        const d = ev.data || {};
        switch (ev.event) {
          case "step":
            confLog("step", d.label || d.name || "");
            break;
          case "page":
            renderConfPage(d);
            break;
          case "log":
            confLog("log", d.line != null ? d.line : String(d));
            break;
          case "done": {
            const summary = "Готово: создано " + (d.synced || 0) +
              ", обновлено " + (d.updated || 0) +
              ", пропущено " + (d.skipped || 0) +
              ", удалено " + (d.deleted || 0) +
              ", вложений " + (d.attachments || 0) +
              ", ошибок " + (d.failed || 0) +
              " · " + (d.duration_s != null ? d.duration_s : "?") + " с";
            confLog("done", summary);
            toast("ok", "Синхронизация завершена", summary);
            loadConfluenceStatus();
            break;
          }
          case "error":
            confLog("error", (d.code ? "[" + d.code + "] " : "") + (d.message || "Ошибка") + (d.detail ? " — " + d.detail : ""));
            toast("err", "Ошибка синхронизации", d.message || d.code || "");
            break;
        }
      });
    } catch (e) {
      confLog("error", "Ошибка: " + e.message);
      toast("err", "Ошибка синхронизации", e.message);
    } finally {
      state.confSyncing = false;
      dom.confSync.disabled = false;
    }
  }

  /* ============================ INDEX MAINTENANCE ============================
   * Reindex (this user's documents, non-destructive) and collection rebuild
   * (destructive, every user). Both are server-side JOBS: the POST returns a
   * jobId immediately and we POLL the status endpoint — the work outlives this
   * page, so reopening the drawer reattaches to whatever is still running
   * instead of starting anything.
   */
  const ADMIN_POLL_MS = 2500;
  const REBUILD_PHASE = {
    dropping: "удаление векторов",
    creating: "создание коллекции",
    indexing: "индексация документов",
    done: "завершение",
  };

  function num(n) {
    if (typeof n !== "number" || !isFinite(n)) return null;
    try { return n.toLocaleString("ru-RU"); } catch (_) { return String(n); }
  }

  function indexLog(node, kind, text) {
    if (!node) return;
    const cls = kind === "error" ? "ln lv-err"
      : kind === "done" ? "step lv-ok"
      : kind === "step" ? "step"
      : "ln lv-dim";
    const line = el("span", cls);
    line.textContent = ((kind === "step" || kind === "done") ? "▸ " : "") + text;
    node.hidden = false;
    node.appendChild(line);
    node.scrollTop = node.scrollHeight;
  }

  // Per-file errors arrive either as plain strings or as {path, error} objects.
  function jobErrorText(e) {
    if (e == null) return "";
    if (typeof e === "string") return e;
    const path = e.path || e.file || e.name || "";
    const msg = e.error || e.message || e.reason || "";
    if (path && msg) return path + " — " + msg;
    if (path || msg) return path || msg;
    try { return JSON.stringify(e); } catch (_) { return String(e); }
  }

  function logJobErrors(node, errors) {
    const list = Array.isArray(errors) ? errors : [];
    list.forEach((e) => indexLog(node, "error", jobErrorText(e)));
  }

  /* ---- collection state + scheme version ---- */
  function renderCollectionState(info) {
    state.collection = info || null;
    if (!info) return;
    const parts = [];
    if (info.collection) parts.push("Коллекция " + info.collection);
    if (info.alias && info.alias !== info.collection) parts.push("алиас " + info.alias);
    const points = num(info.pointsCount);
    if (points != null) {
      parts.push(points + " " + pluralRu(info.pointsCount, "фрагмент", "фрагмента", "фрагментов"));
    }
    dom.indexCollectionState.textContent = parts.join(" · ");

    // Заблокированный индекс: поиска нет вообще. Текст статичен в разметке —
    // подставляем только имя коллекции, его же операторy предстоит ввести ниже.
    const blocked = info.blocked === true;
    if (dom.indexBlocked) dom.indexBlocked.hidden = !blocked;
    if (dom.indexBlockedName) dom.indexBlockedName.textContent = info.collection || "";

    // One sentence about what a scheme mismatch MEANS — not two version numbers.
    // A null schemeVersion (collection carries no marker) is stale too, so the
    // test is "not equal to expected", not "both present and different".
    //
    // Заблокированная коллекция всегда «не той схемы», но говорить про качество
    // поиска по словам, когда поиска нет вовсе, — значит уводить в сторону:
    // сообщение выше уже описало и проблему, и лечение.
    const stale = !blocked
      && info.expectedSchemeVersion != null
      && info.schemeVersion !== info.expectedSchemeVersion;
    dom.indexSchemeWarn.hidden = !stale;
    dom.indexSchemeWarn.textContent = stale
      ? "Индекс собран по устаревшей схеме: поиск по словам работает хуже обычного, пока коллекция не пересоздана."
      : "";
    if (dom.rebuildCollectionName) dom.rebuildCollectionName.textContent = info.collection || "";
    // Отдельное предупреждение в диалоге подтверждения: удаляется единственный
    // существующий индекс, и снимка с него нет.
    if (dom.rebuildLegacyWarn) dom.rebuildLegacyWarn.hidden = !blocked;
    if (dom.rebuildLegacyName) dom.rebuildLegacyName.textContent = info.collection || "";
  }

  function showRebuildUnavailable(message) {
    state.collection = null;
    if (dom.indexBlocked) dom.indexBlocked.hidden = true;
    if (dom.rebuildLegacyWarn) dom.rebuildLegacyWarn.hidden = true;
    dom.rebuildBlock.hidden = false;
    dom.rebuildControls.hidden = true;
    dom.rebuildUnavailable.hidden = false;
    dom.rebuildUnavailable.textContent = message;
  }

  async function loadCollectionInfo() {
    try {
      const info = await apiSend("/api/admin/collection", "GET");
      renderCollectionState(info || {});
      dom.rebuildBlock.hidden = false;
      dom.rebuildControls.hidden = false;
      dom.rebuildUnavailable.hidden = true;
    } catch (e) {
      if (e.handled) return;
      if (e.code === "COLLECTION_API_UNAVAILABLE" || e.status === 501) {
        dom.indexCollectionState.textContent = "";
        dom.indexSchemeWarn.hidden = true;
        showRebuildUnavailable(e.message || "Пересоздание коллекции недоступно в этой версии сервиса");
        return;
      }
      dom.indexCollectionState.textContent = "Не удалось получить состояние индекса: " + e.message;
      // Состояние неизвестно — старая «заблокировано» на экране была бы враньём.
      if (dom.indexBlocked) dom.indexBlocked.hidden = true;
      if (dom.rebuildLegacyWarn) dom.rebuildLegacyWarn.hidden = true;
      dom.rebuildBlock.hidden = true;
    }
  }

  /* ---- vault reindex ---- */
  function setReindexRunning(on) {
    dom.reindexBtn.disabled = on || state.reindexBusy;
    dom.reindexBtn.textContent = on ? "Переиндексация идёт…" : "Переиндексировать вольт";
  }

  function renderReindexProgress(st) {
    const done = num(st.filesProcessed) || 0;
    const total = num(st.totalFiles);
    // `totalFiles` наполняется только по завершении сканирования вольта, а
    // счётчик обработанных растёт с первого же события — отсюда «Обработано 120
    // из 0», которое читается как сломанный счётчик. Пока знаменателя нет,
    // честнее не показывать его вовсе и сказать, что идёт подсчёт.
    const knownTotal = total != null && total > 0;
    const bits = [
      "Обработано " + done + (knownTotal ? " из " + total : "") + " файлов",
    ];
    if (!knownTotal && st.status === "running") bits.push("идёт подсчёт файлов");
    if (st.errorCount) bits.push("ошибок " + st.errorCount);
    if (st.status === "running") bits.push("задача идёт на сервере — окно можно закрыть");
    dom.reindexProgress.hidden = false;
    dom.reindexProgress.textContent = bits.join(" · ");
  }

  /* ---- прогон оценки качества ---- */

  // Раздел показывается, только если харнесс есть в образе: на сборке без него
  // кнопка вела бы в 501, а пустая кнопка хуже отсутствующей.
  async function refreshEvalSection() {
    if (!dom.evalBlock) return;
    let st;
    try {
      st = await apiGet("/api/eval/status");
    } catch (e) {
      dom.evalBlock.hidden = true;
      return;
    }
    dom.evalBlock.hidden = !st.available;
    if (!st.available) return;
    renderEvalDatasets(st.datasets || []);
    renderEvalReports(st.reports || []);
    if (st.job) {
      renderEvalJob(st.job);
      // Прогон переживает перезагрузку страницы — подхватываем его опрос.
      if (st.job.status === "running") pollEval();
    }
  }

  function renderEvalDatasets(datasets) {
    if (!dom.evalDataset) return;
    const selected = dom.evalDataset.value || "builtin:golden.jsonl";
    dom.evalDataset.innerHTML = "";
    datasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      const origin = dataset.source === "upload" ? "загружен" : "в образе";
      option.textContent = dataset.name + " · " + dataset.rows + " вопросов · " + origin;
      option.dataset.rows = String(dataset.rows || 0);
      option.dataset.withGroundTruth = String(dataset.with_ground_truth || 0);
      dom.evalDataset.appendChild(option);
    });
    if ([...dom.evalDataset.options].some((option) => option.value === selected)) {
      dom.evalDataset.value = selected;
    } else if ([...dom.evalDataset.options].some((option) => option.value === "builtin:golden.jsonl")) {
      dom.evalDataset.value = "builtin:golden.jsonl";
    }
    renderEvalDatasetInfo();
  }

  function renderEvalDatasetInfo() {
    if (!dom.evalDatasetInfo || !dom.evalDataset) return;
    const option = dom.evalDataset.selectedOptions[0];
    if (!option) {
      dom.evalDatasetInfo.textContent = "Загрузите JSONL или Excel (.xlsx) с вопросами.";
      return;
    }
    const rows = parseInt(option.dataset.rows || "0", 10);
    const withTruth = parseInt(option.dataset.withGroundTruth || "0", 10);
    dom.evalDatasetInfo.textContent = withTruth < rows
      ? "Эталонные ответы: " + withTruth + " из " + rows + ". Метрики по ground_truth будут неполными."
      : "Эталонные ответы заполнены для всех " + rows + " вопросов.";
  }

  async function uploadEvalDataset(file) {
    if (!file) return;
    const original = dom.evalUploadBtn.textContent;
    dom.evalUploadBtn.disabled = true;
    dom.evalUploadBtn.textContent = "Загружаю…";
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/eval/dataset", {
        method: "POST",
        body: form,
        headers: authHeaders(),
      });
      if (res.status === 401 && isServerMode()) { handleUnauthorized(); return; }
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message = body && body.error && body.error.message;
        throw new Error(message || "HTTP " + res.status);
      }
      const st = await apiGet("/api/eval/status");
      renderEvalDatasets(st.datasets || []);
      dom.evalDataset.value = body.id;
      renderEvalDatasetInfo();
      toast("ok", "Набор загружен", body.rows + " вопросов");
    } catch (e) {
      if (!e.handled) toast("err", "Не удалось загрузить вопросы", e.message);
    } finally {
      dom.evalUploadBtn.disabled = false;
      dom.evalUploadBtn.textContent = original;
      dom.evalFile.value = "";
    }
  }

  function renderEvalReports(names) {
    if (!dom.evalReports) return;
    dom.evalReports.innerHTML = "";
    if (!names.length) { dom.evalReports.hidden = true; return; }
    names.forEach((n) => {
      const b = el("button", "btn ghost");
      b.type = "button";
      b.textContent = "Скачать " + n;
      b.addEventListener("click", () => downloadReport(n, b));
      dom.evalReports.appendChild(b);
    });
    dom.evalReports.hidden = false;
  }

  // Кнопка, а не <a href>: маршрут закрыт bearer-токеном, а браузерная навигация
  // по ссылке шлёт куки и НЕ шлёт Authorization — такая ссылка молча отдавала бы
  // 401 (проверено на живом образе). Поэтому качаем fetch-ем с токеном и
  // отдаём файл через object URL.
  async function downloadReport(name, btn) {
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Скачиваю…";
    try {
      const res = await fetch("/api/eval/report?name=" + encodeURIComponent(name), {
        headers: authHeaders(),
      });
      if (res.status === 401 && isServerMode()) { handleUnauthorized(); return; }
      if (!res.ok) throw new Error("HTTP " + res.status);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Отзываем не сразу: Safari успевает начать загрузку не мгновенно.
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) {
      toast("err", "Не удалось скачать", e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  function renderEvalJob(job) {
    if (!dom.evalProgress) return;
    const mins = Math.floor((job.elapsed_sec || 0) / 60);
    const secs = (job.elapsed_sec || 0) % 60;
    const clock = mins ? mins + " мин " + secs + " с" : secs + " с";
    const bits = ["Прогон «" + job.label + "»"];
    if (job.status === "running") bits.push("идёт " + clock + " — страницу можно закрыть");
    else if (job.status === "completed") bits.push("готов за " + clock);
    else bits.push("не удался за " + clock + (job.error ? ": " + job.error : ""));
    dom.evalProgress.hidden = false;
    dom.evalProgress.textContent = bits.join(" · ");

    // Лог перерисовываем целиком: сервер отдаёт хвост, а не приращение, и
    // склеивать их на клиенте значило бы дублировать строки при каждом опросе.
    if (dom.evalConsole) {
      dom.evalConsole.innerHTML = "";
      (job.log || []).forEach((line) => indexLog(dom.evalConsole, "info", line));
    }
    if (dom.evalBtn) {
      dom.evalBtn.disabled = job.status === "running";
      dom.evalBtn.textContent = job.status === "running" ? "Прогон идёт…" : "Прогнать оценку";
    }
  }

  function stopEvalPoll() {
    if (state.evalTimer) clearTimeout(state.evalTimer);
    state.evalTimer = null;
  }

  async function pollEval() {
    stopEvalPoll();
    let st;
    try {
      st = await apiGet("/api/eval/status");
    } catch (e) {
      if (!e.handled) indexLog(dom.evalConsole, "error", "Статус недоступен: " + e.message);
      return;
    }
    if (!st.job) return;
    renderEvalJob(st.job);
    renderEvalReports(st.reports || []);
    if (st.job.status === "running") {
      // Раз в 5 секунд: прогон идёт минутами, чаще опрашивать незачем.
      state.evalTimer = setTimeout(pollEval, 5000);
    } else {
      toast(st.job.status === "completed" ? "ok" : "err",
            st.job.status === "completed" ? "Прогон завершён" : "Прогон не удался",
            st.job.error || "Отчёты внизу раздела");
    }
  }

  async function startEval() {
    const label = (dom.evalLabel && dom.evalLabel.value.trim()) || "baseline";
    const limit = dom.evalLimit ? parseInt(dom.evalLimit.value, 10) || 0 : 0;
    const dataset = dom.evalDataset ? dom.evalDataset.value : "builtin:golden.jsonl";
    const many = limit === 0 || limit > 10;
    if (many && !confirm(
      "На каждый вопрос приходится несколько обращений к модели. Полный прогон может идти долго. Запустить?"
    )) return;
    if (dom.evalConsole) { dom.evalConsole.innerHTML = ""; dom.evalConsole.hidden = true; }
    try {
      const job = await apiSend("/api/eval/run", "POST", {
        label: label,
        limit: limit,
        dataset: dataset,
      });
      renderEvalJob(job);
      pollEval();
    } catch (e) {
      if (!e.handled) toast("err", "Не удалось запустить", e.message);
    }
  }

  function stopReindexPoll() {
    if (state.reindexTimer) clearTimeout(state.reindexTimer);
    state.reindexTimer = null;
  }

  async function pollReindex() {
    stopReindexPoll();
    let st;
    try {
      const q = state.reindexJob ? "?jobId=" + encodeURIComponent(state.reindexJob) : "";
      st = await apiSend("/api/admin/reindex/status" + q, "GET");
    } catch (e) {
      if (!e.handled) indexLog(dom.reindexConsole, "error", "Не удалось получить статус: " + e.message);
      setReindexRunning(false);
      return;
    }
    st = st || {};
    if (!st.status || st.status === "idle") {
      state.reindexJob = null;
      state.reindexWatching = false;
      setReindexRunning(false);
      return;
    }
    state.reindexJob = st.jobId || state.reindexJob;
    renderReindexProgress(st);
    if (st.status === "running") {
      state.reindexWatching = true;
      setReindexRunning(true);
      state.reindexTimer = setTimeout(pollReindex, ADMIN_POLL_MS);
      return;
    }
    // terminal — announce it only in the tab that actually watched the job run
    setReindexRunning(false);
    state.reindexJob = null;
    if (!state.reindexWatching) return;
    state.reindexWatching = false;
    const summary = "Обработано " + (num(st.filesProcessed) || "0") + " файлов, ошибок " + (st.errorCount || 0);
    logJobErrors(dom.reindexConsole, st.errors);
    if (st.status === "failed") {
      indexLog(dom.reindexConsole, "error", "Переиндексация завершилась с ошибкой. " + summary);
      toast("err", "Переиндексация не удалась", summary);
    } else {
      indexLog(dom.reindexConsole, "done", "Переиндексация завершена. " + summary);
      toast("ok", "Переиндексация завершена", summary);
    }
    loadCollectionInfo();
  }

  async function startReindex() {
    if (state.reindexBusy || state.reindexTimer) return; // double-click guard
    state.reindexBusy = true;
    dom.reindexBtn.disabled = true;
    dom.reindexConsole.hidden = false;
    dom.reindexConsole.textContent = "";
    indexLog(dom.reindexConsole, "step", "Запуск переиндексации…");
    try {
      const res = await apiSend("/api/admin/reindex", "POST", { scope: "full" });
      state.reindexJob = (res && res.jobId) || null;
      if (res && res.attached) indexLog(dom.reindexConsole, "step", "Переиндексация уже шла — подключились к ней.");
    } catch (e) {
      if (!e.handled) {
        indexLog(dom.reindexConsole, "error", e.message);
        toast("err", "Не удалось запустить переиндексацию", e.message);
      }
      state.reindexBusy = false;
      setReindexRunning(false);
      return;
    }
    state.reindexBusy = false;
    state.reindexWatching = true;
    setReindexRunning(true);
    pollReindex();
  }

  /* ---- collection rebuild ---- */
  function setRebuildRunning(on) {
    dom.rebuildBtn.disabled = on || state.rebuildBusy;
    dom.rebuildBtn.textContent = on ? "Пересоздание идёт…" : "Пересоздать коллекцию";
    if (on) closeRebuildConfirm();
  }

  function openRebuildConfirm() {
    if (state.rebuildTimer) return;
    dom.rebuildConfirm.hidden = false;
    dom.rebuildConfirmInput.value = ""; // never pre-filled: it must be typed
    dom.rebuildGo.disabled = true;
    dom.rebuildConfirmInput.focus();
  }

  function closeRebuildConfirm() {
    dom.rebuildConfirm.hidden = true;
    dom.rebuildConfirmInput.value = "";
    dom.rebuildGo.disabled = true;
  }

  function checkRebuildConfirm() {
    const expected = (state.collection && state.collection.collection) || "";
    dom.rebuildGo.disabled = !expected || dom.rebuildConfirmInput.value.trim() !== expected;
  }

  function renderRebuildProgress(st) {
    const bits = [];
    const phase = REBUILD_PHASE[st.phase] || st.phase;
    if (phase) bits.push("Этап: " + phase);
    if (st.usersTotal != null) bits.push("пользователей " + (st.usersDone || 0) + " из " + st.usersTotal);
    const files = num(st.filesProcessed);
    if (files != null) bits.push("файлов " + files);
    if (st.errorCount) bits.push("ошибок " + st.errorCount);
    if (st.status === "running") bits.push("задача идёт на сервере — окно можно закрыть");
    dom.rebuildProgress.hidden = false;
    dom.rebuildProgress.textContent = bits.join(" · ");
  }

  function stopRebuildPoll() {
    if (state.rebuildTimer) clearTimeout(state.rebuildTimer);
    state.rebuildTimer = null;
  }

  async function pollRebuild() {
    stopRebuildPoll();
    let st;
    try {
      const q = state.rebuildJob ? "?jobId=" + encodeURIComponent(state.rebuildJob) : "";
      st = await apiSend("/api/admin/collection/rebuild/status" + q, "GET");
    } catch (e) {
      if (!e.handled) indexLog(dom.rebuildConsole, "error", "Не удалось получить статус: " + e.message);
      setRebuildRunning(false);
      return;
    }
    st = st || {};
    if (!st.status || st.status === "idle") {
      state.rebuildJob = null;
      state.rebuildWatching = false;
      setRebuildRunning(false);
      return;
    }
    state.rebuildJob = st.jobId || state.rebuildJob;
    if (st.phase && st.phase !== state.rebuildPhase) {
      state.rebuildPhase = st.phase;
      indexLog(dom.rebuildConsole, "step", REBUILD_PHASE[st.phase] || st.phase);
    }
    renderRebuildProgress(st);
    if (st.status === "running") {
      state.rebuildWatching = true;
      setRebuildRunning(true);
      state.rebuildTimer = setTimeout(pollRebuild, ADMIN_POLL_MS);
      return;
    }
    setRebuildRunning(false);
    state.rebuildJob = null;
    state.rebuildPhase = null;
    if (!state.rebuildWatching) return;
    state.rebuildWatching = false;
    const summary = "Обработано файлов " + (num(st.filesProcessed) || "0")
      + ", пользователей " + (st.usersDone || 0) + " из " + (st.usersTotal != null ? st.usersTotal : "?")
      + ", ошибок " + (st.errorCount || 0);
    logJobErrors(dom.rebuildConsole, st.errors);
    if (st.status === "failed") {
      indexLog(dom.rebuildConsole, "error", "Пересоздание завершилось с ошибкой. " + summary);
      toast("err", "Пересоздание не удалось", summary);
    } else {
      indexLog(dom.rebuildConsole, "done", "Коллекция пересоздана. " + summary);
      toast("ok", "Коллекция пересоздана", summary);
    }
    loadCollectionInfo();
  }

  async function startRebuild() {
    if (state.rebuildBusy || state.rebuildTimer) return; // double-click guard
    const confirmName = dom.rebuildConfirmInput.value.trim();
    if (!confirmName) return;
    state.rebuildBusy = true;
    dom.rebuildGo.disabled = true;
    dom.rebuildBtn.disabled = true;
    dom.rebuildConsole.hidden = false;
    dom.rebuildConsole.textContent = "";
    state.rebuildPhase = null;
    indexLog(dom.rebuildConsole, "step", "Запуск пересоздания коллекции…");
    try {
      const res = await apiSend("/api/admin/collection/rebuild", "POST", { confirm: confirmName });
      state.rebuildJob = (res && res.jobId) || null;
      if (res && res.attached) indexLog(dom.rebuildConsole, "step", "Пересоздание уже шло — подключились к нему.");
      closeRebuildConfirm();
    } catch (e) {
      if (!e.handled) {
        indexLog(dom.rebuildConsole, "error", e.message);
        toast("err", "Не удалось запустить пересоздание", e.message);
      }
      state.rebuildBusy = false;
      setRebuildRunning(false);
      checkRebuildConfirm();
      return;
    }
    state.rebuildBusy = false;
    state.rebuildWatching = true;
    setRebuildRunning(true);
    pollRebuild();
  }

  // Drawer open: reattach to anything already running (this browser or not) and
  // refresh the collection state. Never starts a job.
  async function loadIndexState() {
    if (!dom.reindexBtn) return;
    await loadCollectionInfo();
    if (!state.reindexTimer) pollReindex();
    if (!state.rebuildTimer && !dom.rebuildControls.hidden) pollRebuild();
  }

  /* ============================ CHAT ============================ */
  function applyRagUI() {
    dom.ragSwitch.setAttribute("aria-checked", String(state.rag));
    dom.ragSublabel.textContent = state.rag
      ? "Поиск по базе знаний включён"
      : "Обычный чат GigaChat";
    dom.limitWrap = dom.limitWrap || document.getElementById("limit-wrap");
    if (dom.limitWrap) dom.limitWrap.style.opacity = state.rag ? "1" : ".5";
  }
  function toggleRag() {
    state.rag = !state.rag;
    applyRagUI();
  }

  function clearWelcome() {
    const w = dom.thread.querySelector(".welcome");
    if (w) w.remove();
  }
  function renderWelcome() {
    dom.thread.textContent = "";
    const w = el("div", "welcome");
    w.appendChild(el("h2", null, "Чем помочь?"));
    w.appendChild(el("p", null, "Задайте вопрос про внутренние регламенты и документы. Ответы формируются на основе вашей базы знаний."));
    const sug = el("div", "suggestions");
    ["Что говорит регламент об отпусках?", "Как оформить командировку?", "Кратко изложи политику информационной безопасности"].forEach((txt) => {
      const b = el("button", "suggestion", txt);
      b.type = "button";
      b.addEventListener("click", () => { dom.input.value = txt; autoGrow(); dom.input.focus(); });
      sug.appendChild(b);
    });
    w.appendChild(sug);
    dom.thread.appendChild(w);
  }

  function addMessageNode(role, contentHtmlOrText, opts) {
    opts = opts || {};
    const msg = el("div", "msg " + role);
    const bubble = el("div", "bubble" + (opts.err ? " err-bubble" : ""));
    bubble.appendChild(el("div", "role", role === "user" ? "Вы" : "Ассистент"));
    const content = el("div", "content");
    if (opts.asHtml) content.innerHTML = contentHtmlOrText;
    else content.textContent = contentHtmlOrText;
    bubble.appendChild(content);
    msg.appendChild(bubble);
    dom.thread.appendChild(msg);
    return { msg, bubble, content };
  }

  const DEPTH_LABEL = { file: "файл", section: "раздел", chunk: "фрагмент" };
  function pluralRu(n, one, few, many) {
    const m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) return few;
    return many;
  }
  function renderSources(bubble, sources, contextChars) {
    if (!sources || !sources.length) return;
    const wrap = el("div", "sources");

    // Collapsed-by-default disclosure. The header carries the count and the
    // existing context-size info; chips are hidden until the header is clicked.
    const count = sources.length;
    let headText = "Источники (" + count + ")";
    if (typeof contextChars === "number" && contextChars > 0) {
      headText += " · контекст " + contextChars.toLocaleString("ru-RU") + " симв.";
    }
    const toggle = el("button", "src-toggle");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");
    const chev = el("span", "src-chev", "▸");
    chev.setAttribute("aria-hidden", "true");
    toggle.appendChild(chev);
    toggle.appendChild(el("span", "src-head", headText));

    const chips = el("div", "chips");
    chips.hidden = true; // collapsed by default (per-render UI state, not persisted)
    sources.forEach((s) => {
      const chip = el("div", "chip");
      const titleText = s.title || s.name || "Документ";
      // Full path is kept only as a tooltip for power users — never shown inline.
      if (s.path) chip.title = s.path;
      // Clickable Confluence source: title becomes a link when a safe http(s)
      // url is provided; otherwise it stays plain text as before.
      if (s.url && /^https?:\/\//i.test(s.url)) {
        const a = el("a", "ct link");
        a.href = s.url; // property assignment — not innerHTML, so injection-safe
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.appendChild(document.createTextNode(titleText));
        a.appendChild(el("span", "cext", "↗"));
        chip.appendChild(a);
      } else {
        chip.appendChild(el("span", "ct", titleText));
      }
      const depthLabel = DEPTH_LABEL[s.depth];
      if (depthLabel) chip.appendChild(el("span", "cd", depthLabel));
      if (s.score != null) {
        const score = typeof s.score === "number" ? s.score.toFixed(3) : String(s.score);
        chip.appendChild(el("span", "cs", score));
      }
      chips.appendChild(chip);
    });

    const setOpen = (open) => {
      toggle.setAttribute("aria-expanded", String(open));
      chips.hidden = !open;
      chev.classList.toggle("open", open);
    };
    toggle.addEventListener("click", () => setOpen(toggle.getAttribute("aria-expanded") !== "true"));

    wrap.appendChild(toggle);
    wrap.appendChild(chips);
    bubble.appendChild(wrap);
  }

  /* ---- 👍/👎 answer feedback (wave 5.4) ----
   * Sits right under the assistant bubble, next to the sources strip. A click
   * POSTs {chat_id, message_index, vote} to /api/feedback (the server appends it
   * to the per-user rag_log.jsonl); the vote is then reflected on the button and
   * both buttons are disabled. State lives ONLY in the DOM — nothing is cached
   * or persisted client-side, so a page reload simply shows fresh buttons.
   * `messageIndex` is the position in state.messages / history.messages, which
   * is exactly what the chat route logs as `message_index`. */
  function renderFeedback(bubble, chatId, messageIndex) {
    if (!bubble || !chatId) return;
    if (!(messageIndex >= 0)) return;
    if (bubble.querySelector(".fb")) return; // idempotent per bubble

    const wrap = el("div", "fb");
    wrap.style.cssText = "display:flex;gap:8px;margin-top:10px";
    const mk = (glyph, label) => {
      const b = el("button", "linkbtn", glyph);
      b.type = "button";
      b.title = label;
      b.setAttribute("aria-label", label);
      b.setAttribute("aria-pressed", "false");
      return b;
    };
    const up = mk("👍", "Хороший ответ");
    const down = mk("👎", "Плохой ответ");
    const buttons = [up, down];
    let sending = false;

    const vote = async (btn, value) => {
      if (sending || btn.getAttribute("aria-pressed") === "true") return;
      sending = true;
      buttons.forEach((b) => { b.disabled = true; });
      try {
        await apiSend("/api/feedback", "POST", {
          chat_id: chatId,
          message_index: messageIndex,
          vote: value,
        });
        btn.setAttribute("aria-pressed", "true");
        btn.style.borderColor = "var(--accent)";
        btn.style.color = "var(--accent)";
        buttons.forEach((b) => { if (b !== btn) b.style.opacity = ".35"; });
      } catch (e) {
        if (!e.handled) toast("warn", "Не удалось отправить оценку", e.message);
        sending = false;
        buttons.forEach((b) => { b.disabled = false; });
      }
    };
    up.addEventListener("click", () => vote(up, "up"));
    down.addEventListener("click", () => vote(down, "down"));

    wrap.appendChild(up);
    wrap.appendChild(down);
    bubble.appendChild(wrap);
  }

  /* ---- inline citation linkifier ----
   * Turns `[Источник N]`, `[Источник 1, 2]`, `[Источники 1, 2, 3]` inside a
   * rendered answer into hyperlinks. Only numbers whose source carries a safe
   * http(s) `url` become links; numbers of link-less sources stay plain text.
   * Numbers the model invented (not in the source list at all) also stay plain
   * text but get a `cite-bad` class and are reported once per message. Runs over
   * text nodes via a TreeWalker that skips <a>/<code>/<pre>, so code and existing
   * links are never touched. Applied once on final render (done / abort /
   * history), not per streaming token. hrefs are set via DOM property, never
   * string-built. */
  function linkifyCitations(container, sources) {
    if (!container) return;
    const urlByN = new Map();
    const validN = new Set();
    (sources || []).forEach((s) => {
      if (!s || s.n == null) return;
      const n = Number(s.n);
      if (!Number.isFinite(n)) return;
      validN.add(n);
      if (s.url && /^https?:\/\//i.test(s.url)) urlByN.set(n, s.url);
    });
    // No early return on an empty source list: an answer WITHOUT sources that
    // still cites "[Источник 1]" is the purest form of the bug we are hunting,
    // so it must be flagged too. The TreeWalker below only visits text nodes
    // that actually contain a citation, so this costs nothing otherwise.

    const hasCite = (t) => /\[\s*Источник(?:и|а)?\s+\d/.test(t);
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        for (let p = node.parentNode; p && p !== container; p = p.parentNode) {
          const tag = p.nodeName;
          if (tag === "A" || tag === "CODE" || tag === "PRE") return NodeFilter.FILTER_REJECT;
        }
        return hasCite(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    });
    const targets = [];
    let node;
    while ((node = walker.nextNode())) targets.push(node);
    const unknownN = new Set();
    targets.forEach((n) => replaceCitationsInText(n, urlByN, validN, unknownN));
    // The only console.* in this file: a citation pointing at a source that was
    // never retrieved is an answer-quality bug we need to see while debugging,
    // but it is not actionable for the user, so no toast — one warn per message.
    if (unknownN.size) {
      const list = Array.from(unknownN).sort((a, b) => a - b);
      console.warn("[cognivault] citations reference unknown sources:", list, "available:", validN.size);
    }
  }

  function replaceCitationsInText(node, urlByN, validN, unknownN) {
    const text = node.nodeValue;
    // g1 = "[Источник " (bracket + word + spaces); g2 = numbers+separators; g3 = "]"
    const re = /(\[\s*Источник(?:и|а)?\s+)(\d+(?:\s*[,;]\s*\d+)*)(\s*\])/g;
    let m, last = 0, frag = null;
    while ((m = re.exec(text))) {
      if (!frag) frag = document.createDocumentFragment();
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      frag.appendChild(document.createTextNode(m[1]));
      // split the numbers portion, keeping separators as text; wrap only digits
      m[2].split(/(\d+)/).forEach((part) => {
        if (!part) return;
        if (!/^\d+$/.test(part)) { // separator (comma / semicolon / spaces)
          frag.appendChild(document.createTextNode(part));
          return;
        }
        const n = Number(part);
        const known = validN.has(n);
        if (!known) unknownN.add(n);
        const url = known ? urlByN.get(n) : undefined;
        if (url) {
          const a = document.createElement("a");
          a.href = url; // DOM property — not string-interpolated
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.className = "cite-link";
          a.appendChild(document.createTextNode(part));
          const ext = document.createElement("span");
          ext.className = "cext";
          ext.textContent = "↗";
          a.appendChild(ext);
          frag.appendChild(a);
        } else if (known) {
          frag.appendChild(document.createTextNode(part));
        } else {
          // hallucinated number: plain text, but tagged so it can be styled later
          const bad = document.createElement("span");
          bad.className = "cite-bad";
          bad.textContent = part;
          frag.appendChild(bad);
        }
      });
      frag.appendChild(document.createTextNode(m[3]));
      last = re.lastIndex;
    }
    if (frag) {
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function renderThread() {
    dom.thread.textContent = "";
    if (!state.messages.length) { renderWelcome(); return; }
    state.messages.forEach((m, i) => {
      if (m.role === "user") {
        addMessageNode("user", m.content);
      } else {
        const { bubble, content } = addMessageNode("assistant", "", { asHtml: true });
        content.innerHTML = renderMarkdown(m.content || "");
        linkifyCitations(content, m.sources);
        renderSources(bubble, m.sources, m.context_chars);
        renderFeedback(bubble, state.chatId, i);
      }
    });
    scrollStreamToBottom();
  }

  function setStreamingUI(on) {
    state.streaming = on;
    if (on) {
      dom.sendBtn.textContent = "Стоп";
      dom.sendBtn.classList.add("stop");
      dom.sendBtn.disabled = false;
    } else {
      dom.sendBtn.textContent = "Отправить";
      dom.sendBtn.classList.remove("stop");
      dom.sendBtn.disabled = false;
    }
  }

  async function sendMessage() {
    if (state.streaming) { // acts as Stop
      if (state.abort) state.abort.abort();
      return;
    }
    const text = dom.input.value.trim();
    if (!text) return;

    clearWelcome();
    // user turn
    state.messages.push({ role: "user", content: text });
    addMessageNode("user", text);
    dom.input.value = "";
    autoGrow();
    scrollStreamToBottom();

    // assistant bubble under construction
    const { bubble, content } = addMessageNode("assistant", "", { asHtml: true });
    const cursor = el("span", "cursor");
    content.appendChild(cursor);
    let answer = "";
    let noticeNode = null;
    // Плашка «Модель думает…» для провайдера без стриминга; снимается первым же
    // токеном и в терминальных ветках ниже.
    let waitNode = null;
    let sourcesData = null;
    let contextChars = null;
    scrollStreamToBottom();

    setStreamingUI(true);
    state.abort = new AbortController();

    const payload = {
      messages: state.messages.map((m) => ({ role: m.role, content: m.content })),
      rag: state.rag,
      chat_id: state.chatId,
    };

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: Object.assign({ "Content-Type": "application/json", Accept: "text/event-stream" }, authHeaders()),
        body: JSON.stringify(payload),
        signal: state.abort.signal,
      });
      if (res.status === 401 && isServerMode()) {
        if (cursor.parentNode) cursor.remove();
        if (bubble.parentNode) bubble.parentNode.remove();
        handleUnauthorized();
        return; // finally still resets streaming UI
      }
      if (!res.ok || !res.body) {
        let msg = "HTTP " + res.status;
        try { const j = await res.json(); if (j && j.error) msg = j.error.message; } catch (_) {}
        throw new Error(msg);
      }

      await consumeSSE(res, (ev) => {
        const d = ev.data || {};
        switch (ev.event) {
          case "meta":
            if (d.chat_id) state.chatId = d.chat_id;
            // Провайдер без потока токенов (KitAI: запрос → опрос → готовый
            // ответ). Мигающий курсор над пустым пузырём читается как «идёт
            // печать» и врёт: печатать нечего, пока не придёт весь ответ.
            if (d.streaming === false) {
              cursor.classList.add("hidden");
              waitNode = el("div", "thinking");
              waitNode.textContent = "Модель думает…";
              content.parentNode.insertBefore(waitNode, content.nextSibling);
            }
            break;
          case "sources":
            sourcesData = d.sources || d.items || (Array.isArray(d) ? d : []);
            if (typeof d.context_chars === "number") contextChars = d.context_chars;
            // render sources strip under the growing bubble (remove old first)
            const oldS = bubble.querySelector(".sources");
            if (oldS) oldS.remove();
            renderSources(bubble, sourcesData, contextChars);
            scrollStreamToBottom();
            break;
          case "notice":
            if (!noticeNode) {
              noticeNode = el("div", "notice");
              content.parentNode.insertBefore(noticeNode, content);
            }
            noticeNode.textContent = d.message || d.text || "Поиск по базе недоступен…";
            toast("warn", "Внимание", noticeNode.textContent);
            break;
          case "token": {
            const txt = d.text != null ? d.text : (typeof d === "string" ? d : "");
            if (waitNode) { waitNode.remove(); waitNode = null; }
            answer += txt;
            cursor.parentNode.insertBefore(document.createTextNode(txt), cursor);
            scrollStreamToBottom();
            break;
          }
          case "done":
            if (waitNode) { waitNode.remove(); waitNode = null; }
            if (cursor.parentNode) cursor.remove();
            // finalize with markdown render, then hyperlink inline citations
            content.innerHTML = renderMarkdown(answer);
            linkifyCitations(content, sourcesData);
            state.messages.push({ role: "assistant", content: answer, sources: sourcesData, context_chars: contextChars, rag: state.rag });
            renderFeedback(bubble, state.chatId, state.messages.length - 1);
            break;
          case "error": {
            if (waitNode) { waitNode.remove(); waitNode = null; }
            if (cursor.parentNode) cursor.remove();
            bubble.classList.add("err-bubble");
            let msg = (d.code ? "[" + d.code + "] " : "") + (d.message || "Ошибка");
            if (d.detail) msg += "\n" + d.detail;
            content.textContent = msg;
            toast("err", "Ошибка", d.message || d.code || "");
            break;
          }
        }
      });
    } catch (e) {
      if (e.name === "AbortError") {
        // normal cancellation — keep partial answer
        if (waitNode) { waitNode.remove(); waitNode = null; }
        if (cursor.parentNode) cursor.remove();
        if (answer) {
          content.innerHTML = renderMarkdown(answer);
          linkifyCitations(content, sourcesData);
          state.messages.push({ role: "assistant", content: answer, sources: sourcesData, context_chars: contextChars, rag: state.rag });
          // Оборванный ответ тоже логируется сервером (truncated=true) — значит
          // его можно оценить.
          renderFeedback(bubble, state.chatId, state.messages.length - 1);
        } else {
          bubble.parentNode && bubble.parentNode.remove();
        }
      } else {
        if (cursor.parentNode) cursor.remove();
        bubble.classList.add("err-bubble");
        content.textContent = "Ошибка соединения: " + e.message;
        toast("err", "Ошибка соединения", e.message);
      }
    } finally {
      setStreamingUI(false);
      state.abort = null;
      loadHistory();
    }
  }

  function autoGrow() {
    dom.input.style.height = "auto";
    dom.input.style.height = Math.min(200, dom.input.scrollHeight) + "px";
  }

  /* ============================ HISTORY ============================ */
  async function loadHistory() {
    let data;
    try { data = await apiGet("/api/history"); } catch (_) { return; }
    const items = (data && Array.isArray(data.chats)) ? data.chats : (Array.isArray(data) ? data : []);
    renderHistory(items);
  }

  function renderHistory(items) {
    dom.chatlist.textContent = "";
    dom.historyCount.textContent = items.length ? String(items.length) : "";
    if (!items.length) {
      const empty = el("div", "rail-empty", "Пока нет сохранённых чатов. Начните новый разговор.");
      dom.chatlist.appendChild(empty);
      return;
    }
    items.forEach((it) => {
      const id = it.id != null ? it.id : it.chat_id;
      const item = el("button", "chatitem" + (id === state.chatId ? " active" : ""));
      item.type = "button";
      item.appendChild(el("span", "t", it.title || "Без названия"));
      const meta = el("div", "m");
      meta.appendChild(el("span", null, relTime(it.updated_at || it.created_at || it.ts)));
      if (it.rag || it.has_rag) meta.appendChild(el("span", "tag", "RAG"));
      item.appendChild(meta);
      item.addEventListener("click", () => openChat(id));

      const del = el("button", "del", "🗑");
      del.type = "button";
      del.setAttribute("aria-label", "Удалить чат");
      del.addEventListener("click", (e) => { e.stopPropagation(); deleteChat(id, it.title); });
      item.appendChild(del);

      dom.chatlist.appendChild(item);
    });
  }

  async function openChat(id) {
    try {
      const data = await apiGet("/api/history/" + encodeURIComponent(id));
      const msgs = (data && (data.messages || data.turns)) || [];
      state.chatId = id;
      state.messages = msgs.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
        context_chars: m.context_chars,
        rag: m.rag,
      }));
      renderThread();
      // reflect active state
      Array.prototype.forEach.call(dom.chatlist.children, (c) => c.classList && c.classList.remove("active"));
      loadHistory();
    } catch (e) {
      toast("err", "Не удалось открыть чат", e.message);
    }
  }

  async function deleteChat(id, title) {
    if (!confirm("Удалить чат «" + (title || "Без названия") + "»?")) return;
    try {
      await apiSend("/api/history/" + encodeURIComponent(id), "DELETE");
      if (id === state.chatId) newChat();
      toast("ok", "Чат удалён", "");
      loadHistory();
    } catch (e) {
      toast("err", "Не удалось удалить", e.message);
    }
  }

  function newChat() {
    if (state.abort) state.abort.abort();
    state.chatId = null;
    state.messages = [];
    renderWelcome();
    dom.input.focus();
    Array.prototype.forEach.call(dom.chatlist.children, (c) => c.classList && c.classList.remove("active"));
  }

  /* ============================ INIT ============================ */
  function wireEvents() {
    // header
    dom.themeBtn.addEventListener("click", cycleTheme);
    dom.settingsBtn.addEventListener("click", openDrawer);
    dom.drawerClose.addEventListener("click", closeDrawer);
    dom.scrim.addEventListener("click", closeDrawer);
    dom.bannerBtn.addEventListener("click", openDrawer);

    // rail
    dom.newchatBtn.addEventListener("click", newChat);

    // composer
    dom.ragSwitch.addEventListener("click", toggleRag);
    dom.sendBtn.addEventListener("click", sendMessage);
    dom.input.addEventListener("input", autoGrow);
    dom.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    // settings §1/§2
    dom.tokenToggle.addEventListener("click", toggleTokenVisibility);
    dom.saveConn.addEventListener("click", saveConnection);
    dom.saveCert.addEventListener("click", saveCertificates);

    // settings §M/§S/§P — model, search and prompt tuning
    if (dom.saveModel) dom.saveModel.addEventListener("click", saveModelSettings);
    if (dom.saveRag) dom.saveRag.addEventListener("click", saveRagSettings);
    if (dom.savePrompts) dom.savePrompts.addEventListener("click", savePromptSettings);
    PROMPT_FIELDS.forEach((f) => {
      const node = fieldNode(f);
      if (node) node.addEventListener("input", () => refreshPromptDirty(f));
      const reset = $(f.resetId);
      if (reset) reset.addEventListener("click", () => resetPrompt(f));
    });

    // upload
    dom.dropzone.addEventListener("click", () => dom.fileInput.click());
    dom.dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); dom.fileInput.click(); }
    });
    dom.fileInput.addEventListener("change", () => handleUploadFile(dom.fileInput.files[0]));
    ["dragenter", "dragover"].forEach((ev) => dom.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dom.dropzone.classList.add("drag");
    }));
    ["dragleave", "dragend"].forEach((ev) => dom.dropzone.addEventListener(ev, () => dom.dropzone.classList.remove("drag")));
    dom.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dom.dropzone.classList.remove("drag");
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      handleUploadFile(f);
    });

    // env
    dom.pipTokenToggle.addEventListener("click", togglePipTokenVisibility);
    dom.saveEnvMirror.addEventListener("click", saveEnvMirror);
    dom.envSetupBtn.addEventListener("click", runEnvSetup);
    dom.envImportBtn.addEventListener("click", importEnv);

    // confluence source — null-guard every lookup so one missing element can
    // never abort the rest of the section's (or later) event wiring.
    if (dom.confAuthBasic) dom.confAuthBasic.addEventListener("click", () => applyConfAuthMode("basic"));
    if (dom.confAuthPat) dom.confAuthPat.addEventListener("click", () => applyConfAuthMode("pat"));
    if (dom.confPasswordToggle) dom.confPasswordToggle.addEventListener("click", toggleConfPasswordVisibility);
    if (dom.confPatToggle) dom.confPatToggle.addEventListener("click", toggleConfPatVisibility);
    if (dom.confAutoSync) dom.confAutoSync.addEventListener("change", applyConfInterval);
    if (dom.confSave) dom.confSave.addEventListener("click", saveConfluenceConfig);
    if (dom.confValidate) dom.confValidate.addEventListener("click", validateConfluence);
    if (dom.confSync) dom.confSync.addEventListener("click", syncConfluence);

    // index maintenance — same null-guard discipline as the Confluence block
    if (dom.reindexBtn) dom.reindexBtn.addEventListener("click", startReindex);
    if (dom.evalBtn) dom.evalBtn.addEventListener("click", startEval);
    if (dom.evalDataset) dom.evalDataset.addEventListener("change", renderEvalDatasetInfo);
    if (dom.evalUploadBtn && dom.evalFile) {
      dom.evalUploadBtn.addEventListener("click", () => dom.evalFile.click());
      dom.evalFile.addEventListener("change", () => uploadEvalDataset(dom.evalFile.files[0]));
    }
    if (dom.rebuildBtn) dom.rebuildBtn.addEventListener("click", openRebuildConfirm);
    if (dom.rebuildCancel) dom.rebuildCancel.addEventListener("click", closeRebuildConfirm);
    if (dom.rebuildGo) dom.rebuildGo.addEventListener("click", startRebuild);
    if (dom.rebuildConfirmInput) {
      dom.rebuildConfirmInput.addEventListener("input", checkRebuildConfirm);
      dom.rebuildConfirmInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !dom.rebuildGo.disabled) { e.preventDefault(); startRebuild(); }
      });
    }

    // server-mode login / logout
    if (dom.loginSubmit) dom.loginSubmit.addEventListener("click", submitLogin);
    if (dom.loginToggle) dom.loginToggle.addEventListener("click", toggleLoginTokenVisibility);
    if (dom.loginToken) dom.loginToken.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); submitLogin(); }
    });
    if (dom.logoutBtn) dom.logoutBtn.addEventListener("click", logout);
  }

  async function init() {
    wireEvents();
    renderWelcome();
    // config first (drives mode + theme + rag defaults + banner). No auth needed
    // for GET /api/config in either mode.
    try {
      state.config = await apiGet("/api/config");
    } catch (e) {
      state.config = { mode: "local", cognivault: {}, gigachat: {}, rag: {}, ui: {} };
      toast("err", "Не удалось загрузить конфигурацию", e.message);
    }

    if (isServerMode()) {
      // Multi-tenant: hide admin/local UI, then gate on a valid token.
      applyServerModeUI();
      await bootServer();
    } else {
      // Local mode: unchanged behavior.
      bindConfigToForm();
      await refreshStatus();
      await loadHistory();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
