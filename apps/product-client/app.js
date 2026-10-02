(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    conversation: $("conversation"),
    emptyState: $("emptyState"),
    emptyTitle: $("emptyTitle"),
    emptyBody: $("emptyBody"),
    powerButton: $("powerButton"),
    powerButtonLabel: $("powerButtonLabel"),
    powerStatusText: $("powerStatusText"),
    composer: $("composer"),
    input: $("messageInput"),
    send: $("sendButton"),
    banner: $("runtimeBanner"),
    bannerText: $("runtimeBannerText"),
    retry: $("runtimeRetry"),
    profileButton: $("profileButton"),
    dialog: $("personalizationDialog"),
    closeProfile: $("closeProfileButton"),
    profileForm: $("profileForm"),
    saveState: $("profileSaveState"),
    preferredName: $("preferredName"),
    language: $("language"),
    responseLength: $("responseLength"),
    conversationStyle: $("conversationStyle"),
    initiative: $("initiative"),
    memoryEnabled: $("memoryEnabled"),
    personalInstruction: $("personalInstruction"),
    connectionDetails: $("connectionDetails"),
    remoteEndpoint: $("remoteEndpoint"),
    remoteToken: $("remoteToken"),
    saveConnection: $("saveConnection"),
    clearConnection: $("clearConnection"),
    identitySubline: $("identitySubline"),
  };

  const strings = {
    vi: {
      off: "Đang tắt",
      starting: "Đang khởi động",
      on: "Đang chạy",
      thinking: "Đang nghĩ",
      error: "Cần chú ý",
      turnOn: "Bật AI",
      turnOff: "Tắt AI",
      emptyOffTitle: "Nolane đang nghỉ.",
      emptyOffBody: "Bật AI khi bạn muốn nói chuyện.",
      emptyOnTitle: "Mình ở đây.",
      emptyOnBody: "Cứ nói điều bạn đang nghĩ.",
      input: "Nhắn cho Nolane…",
      send: "Gửi",
      retry: "Thử lại",
      unavailable: "Runtime chưa sẵn sàng.",
      modelMissing: "Thiếu model hoặc tokenizer của bản phát hành.",
      remoteMissing: "Android chưa được ghép với runtime Nolane.",
      saved: "Đã lưu",
      saveError: "Không thể lưu",
      connectionSaved: "Đã lưu kết nối",
      connectionCleared: "Đã xóa kết nối",
      settingsSubline: "của riêng bạn",
    },
    en: {
      off: "Off",
      starting: "Starting",
      on: "Running",
      thinking: "Thinking",
      error: "Needs attention",
      turnOn: "Start AI",
      turnOff: "Stop AI",
      emptyOffTitle: "Nolane is resting.",
      emptyOffBody: "Start the AI whenever you want to talk.",
      emptyOnTitle: "I'm here.",
      emptyOnBody: "Say whatever is on your mind.",
      input: "Message Nolane…",
      send: "Send",
      retry: "Try again",
      unavailable: "The runtime is not ready.",
      modelMissing: "Release model or tokenizer assets are missing.",
      remoteMissing: "Android is not paired with a Nolane runtime yet.",
      saved: "Saved",
      saveError: "Could not save",
      connectionSaved: "Connection saved",
      connectionCleared: "Connection cleared",
      settingsSubline: "yours",
    },
  };

  let profile = {
    preferred_name: "",
    language: "auto",
    response_length: "balanced",
    conversation_style: "natural",
    initiative: "gentle",
    memory_enabled: true,
    personal_instruction: "",
  };
  let runtime = {
    phase: "off",
    powered: false,
    error: null,
  };
  let messages = [];
  let target = { mode: "browser-mock" };
  let sending = false;
  let renderedEventIds = new Set();
  let firstRender = true;

  const locale = () => {
    if (profile.language === "vi" || profile.language === "en") {
      return profile.language;
    }
    return navigator.language?.toLowerCase().startsWith("vi") ? "vi" : "en";
  };

  const t = (key) => strings[locale()][key] || strings.vi[key] || key;

  const hasTauri = () =>
    Boolean(window.__TAURI__ && window.__TAURI__.core?.invoke);

  const invoke = async (command, args = {}) => {
    if (!hasTauri()) {
      throw new Error("Tauri bridge unavailable");
    }
    return window.__TAURI__.core.invoke(command, args);
  };

  const mock = {
    status: {
      schema: "NOLANE-PRODUCT-RUNTIME-STATUS-V1",
      phase: "off",
      powered: false,
      error: null,
      identity_id: "browser-preview",
      state_version: 0,
      interactions: 0,
      open_threads: 0,
      memory_enabled: true,
      initiative: "gentle",
      model_checkpoint_sha256: "preview",
      device: "preview",
    },
    profile: { ...profile, digest: "preview" },
    messages: [],
  };

  async function mockApi(method, path, body) {
    if (method === "GET" && path === "/v1/status") {
      return { ...mock.status };
    }
    if (method === "GET" && path.startsWith("/v1/history")) {
      return { messages: [...mock.messages] };
    }
    if (method === "GET" && path === "/v1/profile") {
      return { ...mock.profile };
    }
    if (method === "PUT" && path === "/v1/profile") {
      mock.profile = { ...mock.profile, ...body };
      mock.status.memory_enabled = Boolean(mock.profile.memory_enabled);
      mock.status.initiative = mock.profile.initiative;
      return { ...mock.profile };
    }
    if (method === "POST" && path === "/v1/power") {
      mock.status.phase = body.enabled ? "on" : "off";
      mock.status.powered = Boolean(body.enabled);
      return { ...mock.status };
    }
    if (method === "POST" && path === "/v1/chat") {
      if (!mock.status.powered) {
        throw new Error("Nolane AI is not running");
      }
      const now = new Date().toISOString();
      mock.messages.push({
        event_id: "preview-u-" + mock.messages.length,
        at: now,
        role: "user",
        text: body.text,
      });
      mock.messages.push({
        event_id: "preview-a-" + mock.messages.length,
        at: now,
        role: "assistant",
        text:
          locale() === "vi"
            ? "Mình đang ở đây. Bản preview này chỉ kiểm tra giao diện; khi chạy trong app, câu trả lời sẽ đến từ runtime Nolane thật."
            : "I'm here. This browser preview only verifies the interface; in the app, replies come from the real Nolane runtime.",
      });
      return { reply: mock.messages.at(-1).text };
    }
    throw new Error("Unsupported mock route " + method + " " + path);
  }

  async function api(method, path, body = null) {
    if (!hasTauri()) {
      return mockApi(method, path, body);
    }
    const result = await invoke("product_api", {
      method,
      path,
      body,
    });
    if (typeof result === "string") {
      return JSON.parse(result);
    }
    return result;
  }

  async function loadTarget() {
    if (!hasTauri()) {
      target = { mode: "browser-mock", endpoint: "mock://local" };
      els.connectionDetails.hidden = true;
      return;
    }
    target = await invoke("runtime_target");
    els.connectionDetails.hidden = target.mode === "local";
    els.remoteEndpoint.value = target.endpoint || "";
  }

  function applyLocale() {
    document.documentElement.lang = locale();
    els.powerButtonLabel.textContent =
      runtime.phase === "on" || runtime.phase === "thinking"
        ? t("turnOff")
        : t("turnOn");
    els.input.placeholder = t("input");
    els.send.setAttribute("aria-label", t("send"));
    els.retry.textContent = t("retry");
    els.identitySubline.textContent = t("settingsSubline");
    renderRuntime();
  }

  function phaseForUi() {
    if (sending && runtime.phase === "on") return "thinking";
    return runtime.phase || "off";
  }

  function renderRuntime() {
    const phase = phaseForUi();
    els.powerButton.dataset.phase = phase;
    els.powerButton.setAttribute("aria-checked", String(runtime.phase === "on"));
    els.powerButton.setAttribute(
      "aria-label",
      runtime.phase === "on" ? t("turnOff") : t("turnOn"),
    );
    els.powerButtonLabel.textContent =
      runtime.phase === "on" || phase === "thinking" ? t("turnOff") : t("turnOn");

    const statusText = {
      off: t("off"),
      starting: t("starting"),
      on: t("on"),
      thinking: t("thinking"),
      error: t("error"),
    }[phase] || t("off");
    els.powerStatusText.textContent = statusText;

    const enabled = runtime.phase === "on" && !sending;
    els.input.disabled = !enabled;
    els.send.disabled = !enabled || !els.input.value.trim();
    els.powerButton.disabled = runtime.phase === "starting" || sending;

    if (runtime.error) {
      showBanner(humanizeRuntimeError(runtime.error), true);
    } else if (target.mode === "unconfigured") {
      showBanner(t("remoteMissing"), false);
    } else {
      hideBanner();
    }

    if (messages.length === 0) {
      els.emptyState.hidden = false;
      if (runtime.phase === "on") {
        els.emptyTitle.textContent = t("emptyOnTitle");
        els.emptyBody.textContent = t("emptyOnBody");
      } else {
        els.emptyTitle.textContent = t("emptyOffTitle");
        els.emptyBody.textContent = t("emptyOffBody");
      }
    } else {
      els.emptyState.hidden = true;
    }
  }

  function humanizeRuntimeError(error) {
    const value = String(error || "");
    if (
      value.includes("checkpoint") ||
      value.includes("tokenizer") ||
      value.includes("release model")
    ) {
      return t("modelMissing");
    }
    return value || t("unavailable");
  }

  function showBanner(message, retry = true) {
    els.banner.hidden = false;
    els.bannerText.textContent = message;
    els.retry.hidden = !retry;
  }

  function hideBanner() {
    els.banner.hidden = true;
    els.retry.hidden = true;
  }

  function nearBottom() {
    const el = els.conversation;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 130;
  }

  function scrollBottom(force = false) {
    if (force || nearBottom()) {
      requestAnimationFrame(() => {
        els.conversation.scrollTop = els.conversation.scrollHeight;
      });
    }
  }

  function renderMessage(message) {
    const node = document.createElement("article");
    node.className =
      "message " + (message.role === "user" ? "user" : "assistant");
    node.dataset.eventId = message.event_id || "";
    if (message.role !== "user") {
      const mark = document.createElement("div");
      mark.className = "assistant-mark";
      mark.setAttribute("aria-hidden", "true");
      node.appendChild(mark);
    }
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.text;
    node.appendChild(content);
    return node;
  }

  function renderMessages(nextMessages) {
    const shouldFollow = firstRender || nearBottom();
    const nextIds = new Set(
      nextMessages.map((m, index) =>
        m.event_id || [m.role, index, m.text].join("-"),
      ),
    );

    if (
      firstRender ||
      nextMessages.length < messages.length ||
      [...renderedEventIds].some((id) => !nextIds.has(id))
    ) {
      els.conversation
        .querySelectorAll(".message, .thinking-row")
        .forEach((node) => node.remove());
      renderedEventIds = new Set();
    }

    nextMessages.forEach((message, index) => {
      const id =
        message.event_id || [message.role, index, message.text].join("-");
      if (renderedEventIds.has(id)) return;
      els.conversation.appendChild(renderMessage(message));
      renderedEventIds.add(id);
    });

    messages = nextMessages;
    firstRender = false;
    renderThinking();
    renderRuntime();
    if (shouldFollow) scrollBottom(true);
  }

  function renderThinking() {
    els.conversation.querySelectorAll(".thinking-row").forEach((n) => n.remove());
    if (!sending) return;

    const row = document.createElement("div");
    row.className = "thinking-row";
    const mark = document.createElement("div");
    mark.className = "assistant-mark";
    mark.setAttribute("aria-hidden", "true");
    const dots = document.createElement("span");
    dots.className = "thinking-dots";
    dots.setAttribute("aria-label", t("thinking"));
    dots.innerHTML = "<i></i><i></i><i></i>";
    row.append(mark, dots);
    els.conversation.appendChild(row);
    scrollBottom(true);
  }

  async function refreshStatus() {
    try {
      runtime = await api("GET", "/v1/status");
      renderRuntime();
    } catch (error) {
      runtime = {
        ...runtime,
        phase: "error",
        powered: false,
        error: String(error?.message || error),
      };
      renderRuntime();
    }
  }

  async function refreshHistory() {
    try {
      const result = await api("GET", "/v1/history");
      renderMessages(Array.isArray(result.messages) ? result.messages : []);
    } catch (error) {
      showBanner(String(error?.message || error), true);
    }
  }

  async function refreshProfile() {
    try {
      profile = await api("GET", "/v1/profile");
      fillProfile();
      applyLocale();
    } catch (error) {
      showBanner(String(error?.message || error), true);
    }
  }

  async function togglePower() {
    const next = runtime.phase !== "on";
    runtime = { ...runtime, phase: next ? "starting" : "off", error: null };
    renderRuntime();
    try {
      runtime = await api("POST", "/v1/power", { enabled: next });
      renderRuntime();
      if (runtime.phase === "on") {
        els.input.focus();
      }
    } catch (error) {
      runtime = {
        ...runtime,
        phase: "error",
        powered: false,
        error: String(error?.message || error),
      };
      renderRuntime();
    }
  }

  function autoResize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 148) + "px";
    els.send.disabled =
      runtime.phase !== "on" || sending || !els.input.value.trim();
  }

  async function sendMessage() {
    const text = els.input.value.trim();
    if (!text || sending || runtime.phase !== "on") return;
    sending = true;
    const pending = {
      event_id: "pending-" + Date.now(),
      at: new Date().toISOString(),
      role: "user",
      text,
    };
    renderMessages([...messages, pending]);
    els.input.value = "";
    autoResize();
    renderRuntime();

    try {
      await api("POST", "/v1/chat", { text });
      await refreshHistory();
    } catch (error) {
      showBanner(String(error?.message || error), true);
      await refreshHistory();
    } finally {
      sending = false;
      renderThinking();
      await refreshStatus();
      renderRuntime();
      els.input.focus();
    }
  }

  function fillProfile() {
    els.preferredName.value = profile.preferred_name || "";
    els.language.value = profile.language || "auto";
    els.responseLength.value = profile.response_length || "balanced";
    els.conversationStyle.value = profile.conversation_style || "natural";
    els.initiative.value = profile.initiative || "gentle";
    els.memoryEnabled.checked = profile.memory_enabled !== false;
    els.personalInstruction.value = profile.personal_instruction || "";
  }

  function profilePatch() {
    return {
      preferred_name: els.preferredName.value,
      language: els.language.value,
      response_length: els.responseLength.value,
      conversation_style: els.conversationStyle.value,
      initiative: els.initiative.value,
      memory_enabled: els.memoryEnabled.checked,
      personal_instruction: els.personalInstruction.value,
    };
  }

  async function saveProfile() {
    els.saveState.textContent = "";
    try {
      profile = await api("PUT", "/v1/profile", profilePatch());
      fillProfile();
      applyLocale();
      els.saveState.textContent = t("saved");
      setTimeout(() => {
        if (els.saveState.textContent === t("saved")) {
          els.saveState.textContent = "";
        }
      }, 1400);
    } catch (error) {
      els.saveState.textContent = t("saveError");
      showBanner(String(error?.message || error), true);
    }
  }

  async function saveConnection() {
    if (!hasTauri()) return;
    const endpoint = els.remoteEndpoint.value.trim();
    const token = els.remoteToken.value;
    try {
      target = await invoke("configure_remote", { endpoint, token });
      els.remoteToken.value = "";
      els.saveState.textContent = t("connectionSaved");
      await refreshAll();
    } catch (error) {
      els.saveState.textContent = t("saveError");
      showBanner(String(error?.message || error), false);
    }
  }

  async function clearConnection() {
    if (!hasTauri()) return;
    try {
      target = await invoke("clear_remote");
      els.remoteEndpoint.value = "";
      els.remoteToken.value = "";
      els.saveState.textContent = t("connectionCleared");
      await refreshAll();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    }
  }

  async function refreshAll() {
    await loadTarget();
    if (target.mode === "unconfigured") {
      runtime = {
        phase: "error",
        powered: false,
        error: null,
      };
      renderRuntime();
      return;
    }
    await Promise.all([
      refreshStatus(),
      refreshHistory(),
      refreshProfile(),
    ]);
  }

  els.powerButton.addEventListener("click", togglePower);
  els.retry.addEventListener("click", refreshAll);

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage();
  });

  els.input.addEventListener("input", autoResize);
  els.input.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing &&
      window.innerWidth > 640
    ) {
      event.preventDefault();
      void sendMessage();
    }
  });

  els.profileButton.addEventListener("click", () => {
    fillProfile();
    els.saveState.textContent = "";
    els.dialog.showModal();
    requestAnimationFrame(() => els.preferredName.focus());
  });

  els.closeProfile.addEventListener("click", () => els.dialog.close());

  els.profileForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void saveProfile().then(() => els.dialog.close());
  });

  els.saveConnection.addEventListener("click", () => void saveConnection());
  els.clearConnection.addEventListener("click", () => void clearConnection());

  els.dialog.addEventListener("click", (event) => {
    const rect = els.dialog.getBoundingClientRect();
    const outside =
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom;
    if (outside) els.dialog.close();
  });

  window.addEventListener("focus", () => void refreshAll());

  autoResize();
  void refreshAll();

  setInterval(() => {
    if (!document.hidden && !sending && target.mode !== "unconfigured") {
      void refreshStatus();
      void refreshHistory();
    }
  }, 3000);
})();
