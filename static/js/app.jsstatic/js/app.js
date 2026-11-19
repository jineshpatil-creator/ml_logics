(() => {
  const form = document.getElementById("form");
  const submit = document.getElementById("submit");
  const banner = document.getElementById("banner");
  const result = document.getElementById("result");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const setErr = (name, msg) => {
    const f = form.querySelector(`.field[data-name="${name}"]`);
    if (!f) return;
    f.classList.toggle("invalid", !!msg);
    f.querySelector("input").setAttribute("aria-invalid", msg ? "true" : "false");
    f.querySelector(".err").textContent = msg || "";
  };
  const clearAll = () => {
    form.querySelectorAll(".field").forEach(f => setErr(f.dataset.name, ""));
    banner.hidden = true;
  };
  const showBanner = msg => { banner.textContent = msg; banner.hidden = false; };

  // clear an error as soon as the user edits that field
  form.addEventListener("input", e => e.target.name && setErr(e.target.name, ""));
  form.addEventListener("reset", () => { clearAll(); result.hidden = true; result.classList.remove("show"); });

  function validateLocally(data) {
    const errs = {};
    form.querySelectorAll(".field").forEach(f => {
      const input = f.querySelector("input"), label = f.querySelector("label").firstChild.textContent.trim();
      const v = data[input.name];
      if (v === "") errs[input.name] = `${label} is required.`;
      else if (!Number.isFinite(Number(v))) errs[input.name] = `${label} must be a number.`;
      else if (input.step === "1" && !Number.isInteger(Number(v))) errs[input.name] = `${label} must be a whole-number code.`;
    });
    return errs;
  }

  function render(data) {
    document.getElementById("class-code").textContent = data.predicted_class;
    document.getElementById("class-echo").textContent = data.predicted_class;
    const bars = document.getElementById("bars");
    bars.innerHTML = "";
    const probs = data.probabilities || [];
    const max = Math.max(...probs.map(p => p.probability), 0);
    probs.forEach(p => {
      const li = document.createElement("li");
      if (p.probability === max) li.className = "top";
      li.innerHTML = `<span>Class ${p.class}</span><span class="track"><span class="fill"></span></span><span>${(p.probability * 100).toFixed(1)}%</span>`;
      bars.appendChild(li);
      const fill = li.querySelector(".fill"), w = (p.probability * 100) + "%";
      reduce ? (fill.style.width = w) : requestAnimationFrame(() => requestAnimationFrame(() => fill.style.width = w));
    });
    if (!probs.length) bars.innerHTML = "<li>Probabilities are not available for this model.</li>";
    result.hidden = false;
    result.classList.remove("show"); void result.offsetWidth; result.classList.add("show");
    result.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }

  form.addEventListener("submit", async e => {
    e.preventDefault();
    clearAll();
    const data = Object.fromEntries(new FormData(form).entries());
    const errs = validateLocally(data);
    if (Object.keys(errs).length) {
      Object.entries(errs).forEach(([k, m]) => setErr(k, m));
      showBanner("Please fix the highlighted fields.");
      form.querySelector(".invalid input").focus();
      return;
    }
    submit.disabled = true; submit.classList.add("loading");
    try {
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (body.fields) Object.entries(body.fields).forEach(([k, m]) => setErr(k, m));
        showBanner(body.error || "The request failed. Please try again.");
        const bad = form.querySelector(".invalid input"); if (bad) bad.focus();
        return;
      }
      render(body);
    } catch {
      showBanner("Could not reach the server. Check your connection and try again.");
    } finally {
      submit.disabled = false; submit.classList.remove("loading");
    }
  });
})();
