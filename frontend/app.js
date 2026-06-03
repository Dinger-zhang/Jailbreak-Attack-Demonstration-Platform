const state = {
  methods: [],
  datasets: [],
  selectedDatasetId: "",
  selectedRow: null,
  selectedJobId: "",
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.remove("show"), 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
      // Keep the HTTP status text.
    }
    throw new Error(message);
  }
  return response.json();
}

function optionList(select, values, preferred) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === preferred) option.selected = true;
    select.appendChild(option);
  });
}

function statusClass(status) {
  return `status ${status || "idle"}`;
}

function truncate(text, size = 220) {
  const value = String(text || "");
  return value.length > size ? `${value.slice(0, size)}...` : value;
}

function setRunningControls(isRunning) {
  $("#startBtn").disabled = isRunning;
  $("#cancelBtn").disabled = !isRunning || !state.selectedJobId;
}

async function loadHealth() {
  const health = await api("/api/health");
  const dot = $("#healthDot");
  dot.className = `signal-dot ${health.ok ? "ok" : "bad"}`;
  $("#healthTitle").textContent = health.ok ? "环境可运行" : "环境需要处理";
  $("#healthText").textContent = health.ok
    ? `后端 Python: ${health.python}`
    : "TAP 目录或依赖不完整，请查看环境检查。";

  if (health.environment.CUSTOM_API_URL) $("#customApiUrl").value = health.environment.CUSTOM_API_URL;
  if (health.environment.CUSTOM_MODEL_NAME) $("#customModelName").value = health.environment.CUSTOM_MODEL_NAME;
  if (!$("#customApiUrl").value) $("#customApiUrl").value = localStorage.getItem("customApiUrl") || "";
  if (!$("#customModelName").value) $("#customModelName").value = localStorage.getItem("customModelName") || "";

  const envItems = {
    TAP: health.tap_found,
    OPENAI_API_KEY: health.environment.OPENAI_API_KEY,
    CUSTOM_API_TOKEN: health.environment.CUSTOM_API_TOKEN,
    ...health.dependencies,
  };
  $("#healthGrid").innerHTML = Object.entries(envItems)
    .map(([key, value]) => `<div class="health-item"><span>${key}</span><b class="${value ? "" : "bad"}">${value ? "OK" : "MISSING"}</b></div>`)
    .join("");
}

async function loadMethods() {
  const payload = await api("/api/methods");
  state.methods = payload.methods;
  const tap = payload.methods.find((item) => item.id === "tap");
  if (!tap) return;
  optionList($("#attackModel"), tap.attack_models, "custom-api-model");
  optionList($("#targetModel"), tap.target_models, "custom-api-model");
  optionList($("#evaluatorModel"), tap.evaluator_models, "custom-api-model");
}

async function loadDatasets() {
  const payload = await api("/api/datasets");
  state.datasets = payload.datasets;
  const select = $("#datasetSelect");
  const current = select.value;
  select.innerHTML = '<option value="">手动输入，不使用数据集</option>';
  state.datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.name} (${dataset.rows} rows)`;
    select.appendChild(option);
  });
  if (current && state.datasets.some((item) => item.id === current)) {
    select.value = current;
  }
}

async function loadDatasetRows(datasetId) {
  state.selectedDatasetId = datasetId;
  state.selectedRow = null;
  $("#datasetRows").innerHTML = "";
  if (!datasetId) {
    $("#datasetMeta").textContent = "选择数据集后会显示字段和样例。";
    return;
  }
  const payload = await api(`/api/datasets/${datasetId}/rows?limit=30`);
  const dataset = payload.dataset;
  $("#datasetMeta").textContent = `字段: ${dataset.columns.join(", ") || "无"} | 总行数: ${dataset.rows}`;
  $("#datasetRows").innerHTML = payload.rows
    .map((row) => `
      <article class="row-card" data-index="${row.index}">
        <strong>#${row.index} ${row.category ? `[${row.category}]` : ""}</strong>
        <p>${escapeHtml(truncate(row.goal || JSON.stringify(row.raw), 180))}</p>
        <p><b>target_str:</b> ${escapeHtml(truncate(row.target_str || "未提供", 120))}</p>
      </article>
    `)
    .join("");

  $("#datasetRows").querySelectorAll(".row-card").forEach((card) => {
    card.addEventListener("click", () => {
      const index = Number(card.dataset.index);
      state.selectedRow = payload.rows.find((row) => row.index === index);
      $("#datasetRows").querySelectorAll(".row-card").forEach((item) => item.classList.remove("selected"));
      card.classList.add("selected");
      if (state.selectedRow.goal) $("#goal").value = state.selectedRow.goal;
      if (state.selectedRow.target_str) $("#targetStr").value = state.selectedRow.target_str;
      if (state.selectedRow.category) $("#category").value = state.selectedRow.category;
      $("#index").value = state.selectedRow.index;
      toast(`已选择数据集行 #${state.selectedRow.index}`);
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readNumber(id) {
  return Number($(`#${id}`).value);
}

function buildRunPayload() {
  const customApiUrl = $("#customApiUrl").value.trim();
  const customModelName = $("#customModelName").value.trim();
  localStorage.setItem("customApiUrl", customApiUrl);
  localStorage.setItem("customModelName", customModelName);

  const payload = {
    method: $("#method").value,
    goal: $("#goal").value.trim(),
    target_str: $("#targetStr").value.trim(),
    attack_model: $("#attackModel").value,
    target_model: $("#targetModel").value,
    evaluator_model: $("#evaluatorModel").value,
    custom_api_url: customApiUrl,
    custom_model_name: customModelName,
    custom_api_token: $("#customApiToken").value.trim(),
    depth: readNumber("depth"),
    width: readNumber("width"),
    branching_factor: readNumber("branchingFactor"),
    n_streams: readNumber("nStreams"),
    keep_last_n: readNumber("keepLastN"),
    attack_max_n_tokens: readNumber("attackMaxTokens"),
    target_max_n_tokens: readNumber("targetMaxTokens"),
    evaluator_max_n_tokens: readNumber("evaluatorMaxTokens"),
    evaluator_temperature: readNumber("evaluatorTemp"),
    max_n_attack_attempts: readNumber("maxAttackAttempts"),
    category: $("#category").value.trim() || "manual",
    index: readNumber("index"),
  };

  if (state.selectedDatasetId && state.selectedRow) {
    payload.dataset_id = state.selectedDatasetId;
    payload.dataset_row_index = state.selectedRow.index;
  }
  return payload;
}

async function startJob() {
  const payload = buildRunPayload();
  const response = await api("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.selectedJobId = response.job.id;
  toast(`任务已启动: ${response.job.id}`);
  watchJob(response.job.id);
  loadJobs();
}

async function cancelCurrentJob() {
  if (!state.selectedJobId) return;
  await api(`/api/jobs/${state.selectedJobId}/cancel`, { method: "POST" });
  toast("已请求取消任务");
  refreshJob(state.selectedJobId);
}

function watchJob(jobId) {
  state.selectedJobId = jobId;
  clearInterval(state.pollTimer);
  refreshJob(jobId);
  state.pollTimer = setInterval(() => refreshJob(jobId), 2200);
}

async function refreshJob(jobId) {
  const [{ job }, logs] = await Promise.all([
    api(`/api/jobs/${jobId}`),
    api(`/api/jobs/${jobId}/logs?tail=500`),
  ]);
  $("#currentJob").textContent = job.id;
  $("#jobStatus").textContent = job.status;
  $("#jobStatus").className = statusClass(job.status);
  $("#logBox").textContent = logs.lines.length ? logs.lines.join("\n") : "暂无日志。";
  $("#logBox").scrollTop = $("#logBox").scrollHeight;

  const running = ["queued", "running", "cancelling"].includes(job.status);
  setRunningControls(running);

  const results = await api(`/api/jobs/${jobId}/results?limit=80`);
  renderResults(results);
  if (!running) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    loadJobs();
  }
}

function renderResults(results) {
  const summary = $("#resultSummary");
  const rows = $("#resultRows");
  if (!results.available) {
    summary.className = "result-summary empty";
    summary.textContent = results.message || "暂无结果。";
    rows.innerHTML = "";
    return;
  }

  const info = results.summary;
  summary.className = "result-summary";
  summary.innerHTML = `
    <strong>结果行数:</strong> ${info.total_rows}
    | <strong>最高评分:</strong> ${info.max_judge_score ?? "N/A"}
    | <strong>成功数:</strong> ${info.success_count ?? 0}
    | <a href="/api/jobs/${results.job.id}/results.csv">下载 CSV</a>
    ${info.best ? `<pre>${escapeHtml(JSON.stringify(info.best, null, 2))}</pre>` : ""}
  `;

  rows.innerHTML = results.rows
    .map((row, index) => `
      <article class="result-card">
        <strong>#${index + 1} | score=${row.judge_scores ?? "N/A"} | iter=${row.iter ?? "N/A"}</strong>
        <p><b>Prompt:</b> ${escapeHtml(truncate(row.prompt, 320))}</p>
        <p><b>Target response:</b> ${escapeHtml(truncate(row.target_response, 320))}</p>
      </article>
    `)
    .join("");
}

async function loadJobs() {
  const payload = await api("/api/jobs");
  $("#jobsList").innerHTML = payload.jobs.length
    ? payload.jobs
        .map((job) => `
          <article class="job-card" data-id="${job.id}">
            <strong>${job.id}</strong>
            <span class="${statusClass(job.status)}">${job.status}</span>
            <p>${escapeHtml(truncate(job.config?.goal || "", 150))}</p>
          </article>
        `)
        .join("")
    : "<p>暂无历史任务。</p>";

  $("#jobsList").querySelectorAll(".job-card").forEach((card) => {
    card.addEventListener("click", () => watchJob(card.dataset.id));
  });
}

async function uploadDataset(event) {
  event.preventDefault();
  const file = $("#datasetFile").files[0];
  if (!file) {
    toast("请选择 CSV、JSONL、JSON 或 TXT 文件");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  const response = await api("/api/datasets", { method: "POST", body: form });
  toast(`数据集已上传: ${response.dataset.name}`);
  await loadDatasets();
  $("#datasetSelect").value = response.dataset.id;
  await loadDatasetRows(response.dataset.id);
}

function bindEvents() {
  $("#uploadForm").addEventListener("submit", (event) => uploadDataset(event).catch((error) => toast(error.message)));
  $("#datasetSelect").addEventListener("change", (event) => loadDatasetRows(event.target.value).catch((error) => toast(error.message)));
  $("#startBtn").addEventListener("click", () => startJob().catch((error) => toast(error.message)));
  $("#cancelBtn").addEventListener("click", () => cancelCurrentJob().catch((error) => toast(error.message)));
}

async function boot() {
  bindEvents();
  await Promise.all([loadHealth(), loadMethods(), loadDatasets(), loadJobs()]);
}

boot().catch((error) => toast(error.message));
