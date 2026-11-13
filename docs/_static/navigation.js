// Keep the theme's native disclosure controls; add keyboard-accessible bulk actions.
document.addEventListener("DOMContentLoaded", () => {
  const nav = document.querySelector(".bd-docs-nav");
  if (!nav || !nav.querySelector("details, input.toctree-checkbox")) return;

  const controls = document.createElement("div");
  controls.className = "pygsti-navigation-controls";
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", "Documentation navigation");
  for (const [label, open] of [["Expand all", true], ["Collapse all", false]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-sm";
    button.textContent = label;
    button.addEventListener("click", () => {
      nav.querySelectorAll("details").forEach((item) => { item.open = open; });
      // Earlier supported theme releases use checkboxes instead of <details>.
      nav.querySelectorAll("input.toctree-checkbox").forEach((item) => {
        item.checked = open;
        item.dispatchEvent(new Event("change", { bubbles: true }));
      });
    });
    controls.append(button);
  }
  nav.prepend(controls);
});
