from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import streamlit as st
import streamlit.components.v2 as components_v2


@dataclass(frozen=True)
class FloatingNavItem:
    key: str
    label: str
    icon: str = ""


def _hex_to_rgba(value: str | None, alpha: float, fallback: str) -> str:
    text = str(value or "").strip()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) == 6:
            try:
                red = int(hex_value[0:2], 16)
                green = int(hex_value[2:4], 16)
                blue = int(hex_value[4:6], 16)
                clamped_alpha = min(max(alpha, 0.0), 1.0)
                return f"rgba({red}, {green}, {blue}, {clamped_alpha:.3f})"
            except ValueError:
                return fallback
    return fallback


def _toolbar_status_label(status: Mapping[str, Any]) -> str:
    state = str(status.get("state", "stopped")).lower()
    if state == "running":
        return "Run"
    if state == "paused":
        return "Pause"
    return "Stop"


def _theme_toggle_svg(theme_mode: str) -> str:
    if theme_mode == "dark":
        return """
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" focusable="false">
          <circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" stroke-width="1.8"></circle>
          <path d="M12 2.8v2.2M12 19v2.2M4.8 12H2.6M21.4 12h-2.2M5.9 5.9l1.6 1.6M16.5 16.5l1.6 1.6M18.1 5.9l-1.6 1.6M7.5 16.5l-1.6 1.6" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"></path>
        </svg>
        """.strip()
    return """
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" focusable="false">
      <path d="M20.2 14.4A8.3 8.3 0 1 1 9.6 3.8a7.2 7.2 0 1 0 10.6 10.6Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"></path>
      <circle cx="17.8" cy="6.2" r="1.1" fill="currentColor"></circle>
    </svg>
    """.strip()


_FLOATING_NAV_HTML = """
<div id="alt-floating-nav-root"></div>
"""


_FLOATING_NAV_CSS = """
:host {
  position: relative;
  display: block;
  height: 0;
  overflow: visible;
  z-index: 40;
}

#alt-floating-nav-root {
  position: relative;
  z-index: 40;
}

.alt-floating-nav {
  position: fixed;
  top: calc(env(safe-area-inset-top, 0px) + var(--alt-nav-top-offset, 0.42rem));
  left: 50%;
  transform: translateX(-50%) translateY(0);
  width: fit-content;
  max-width: calc(100vw - 2rem);
  z-index: 120;
  opacity: 1;
  pointer-events: auto;
  transition:
    opacity 220ms ease,
    transform 220ms ease,
    filter 220ms ease;
  will-change: opacity, transform;
}

.alt-floating-nav.is-hidden {
  opacity: 0;
  transform: translateX(-50%) translateY(-20px);
  pointer-events: none;
  filter: saturate(0.9);
}

.alt-floating-nav-shell {
  display: flex;
  align-items: center;
  gap: 0.34rem;
  min-height: 2.86rem;
  max-width: min(860px, calc(100vw - 1.4rem));
  padding: 0.24rem 0.42rem 0.24rem 0.98rem;
  border-radius: 999px;
  background:
    linear-gradient(135deg, var(--alt-nav-surface, rgba(15, 23, 42, 0.92)), var(--alt-nav-surface-2, rgba(30, 41, 59, 0.82))),
    var(--alt-nav-surface, rgba(15, 23, 42, 0.86));
  border: 1px solid var(--alt-nav-border, rgba(148, 163, 184, 0.22));
  box-shadow: var(
    --alt-nav-shadow,
    0 18px 42px rgba(15, 23, 42, 0.22),
    0 8px 18px rgba(15, 23, 42, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.08)
  );
  backdrop-filter: blur(20px) saturate(145%);
  -webkit-backdrop-filter: blur(20px) saturate(145%);
}

.alt-floating-nav-brand {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  max-width: 0;
  opacity: 0;
  overflow: hidden;
  transform: translateX(-10px);
  transition:
    max-width 220ms ease,
    opacity 220ms ease,
    transform 220ms ease,
    margin-right 220ms ease;
  margin-right: 0;
}

.alt-floating-nav.show-brand .alt-floating-nav-brand {
  max-width: 4.5rem;
  opacity: 1;
  transform: translateX(0);
  margin-right: 0.82rem;
}

.alt-floating-nav-brandmark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 2.16rem;
  padding: 0 0.34rem;
  font-size: 0.94rem;
  font-weight: 900;
  letter-spacing: -0.05em;
  color: var(--alt-nav-text-strong, rgba(255, 255, 255, 0.98));
}

.alt-floating-nav-track {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.18rem;
  flex: 1 1 auto;
  min-width: 0;
  max-width: min(62vw, 42rem);
  overflow-x: auto;
  overflow-y: visible;
  scrollbar-width: none;
  padding: 0.05rem;
}

.alt-floating-nav-track::-webkit-scrollbar {
  display: none;
}

.alt-floating-nav-indicator {
  position: absolute;
  top: 0.1rem;
  left: 0;
  height: calc(100% - 0.2rem);
  width: 0;
  border-radius: 999px;
  background:
    linear-gradient(135deg, rgba(96, 165, 250, 0.28), rgba(59, 130, 246, 0.16)),
    rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(96, 165, 250, 0.3);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.09),
    0 10px 22px rgba(37, 99, 235, 0.18);
  transform: translateX(0);
  transition:
    transform 240ms cubic-bezier(0.22, 1, 0.36, 1),
    width 240ms cubic-bezier(0.22, 1, 0.36, 1),
    opacity 180ms ease;
  pointer-events: none;
  opacity: 1;
}

.alt-floating-nav-btn {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.38rem;
  min-height: 2.08rem;
  padding: 0.42rem 0.76rem;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--alt-nav-text, rgba(241, 245, 249, 0.92));
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  white-space: nowrap;
  cursor: pointer;
  transition:
    transform 150ms ease,
    color 150ms ease,
    background-color 150ms ease,
    opacity 150ms ease;
  flex: 0 0 auto;
}

.alt-floating-nav-btn:hover {
  transform: translateY(-1px);
  color: var(--alt-nav-text-strong, rgba(255, 255, 255, 0.98));
  background: var(--alt-nav-hover, rgba(255, 255, 255, 0.06));
}

.alt-floating-nav-btn:active {
  transform: translateY(0);
}

.alt-floating-nav-btn:focus-visible {
  outline: 2px solid var(--alt-nav-primary, #3b82f6);
  outline-offset: 2px;
}

.alt-floating-nav-btn[aria-current="page"] {
  color: var(--alt-nav-text-strong, rgba(255, 255, 255, 0.99));
}

.alt-floating-nav-icon {
  font-size: 0.76rem;
  opacity: 0.9;
}

.alt-floating-nav-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  min-height: 1.96rem;
  padding: 0.34rem 0.58rem;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  border: 1px solid transparent;
  white-space: nowrap;
}

.alt-floating-nav-meta {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  flex: 0 0 auto;
}

.alt-floating-nav-theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  min-width: 1.96rem;
  min-height: 1.96rem;
  border: 0;
  border-radius: 999px;
  background: var(--alt-nav-chip, rgba(255, 255, 255, 0.06));
  color: var(--alt-nav-text-strong, rgba(255, 255, 255, 0.98));
  cursor: pointer;
  transition:
    transform 150ms ease,
    background-color 150ms ease,
    color 150ms ease;
}

.alt-floating-nav-theme-toggle:hover {
  transform: translateY(-1px);
  background: var(--alt-nav-chip-hover, rgba(255, 255, 255, 0.11));
}

.alt-floating-nav-theme-toggle:focus-visible {
  outline: 2px solid var(--alt-nav-primary, #3b82f6);
  outline-offset: 2px;
}

.alt-floating-nav-status.running {
  background: rgba(16, 185, 129, 0.12);
  border-color: rgba(16, 185, 129, 0.24);
  color: #34d399;
}

.alt-floating-nav-status.paused {
  background: rgba(245, 158, 11, 0.12);
  border-color: rgba(245, 158, 11, 0.24);
  color: #fbbf24;
}

.alt-floating-nav-status.stopped {
  background: rgba(239, 68, 68, 0.11);
  border-color: rgba(239, 68, 68, 0.22);
  color: #fca5a5;
}

@media (max-width: 1024px) {
  .alt-floating-nav-shell {
    min-height: 2.72rem;
    padding: 0.22rem 0.34rem 0.22rem 0.84rem;
  }

  .alt-floating-nav-track {
    max-width: min(58vw, 34rem);
  }

  .alt-floating-nav-btn {
    min-height: 2rem;
    padding: 0.4rem 0.68rem;
    font-size: 0.76rem;
  }
}

@media (max-width: 768px) {
  .alt-floating-nav {
    max-width: calc(100vw - 0.8rem);
    top: calc(env(safe-area-inset-top, 0px) + var(--alt-nav-top-offset-mobile, 0.28rem));
  }

  .alt-floating-nav-shell {
    gap: 0.24rem;
    min-height: 2.5rem;
    padding: 0.18rem 0.24rem 0.18rem 0.64rem;
    border-radius: 1.05rem;
  }

  .alt-floating-nav.show-brand .alt-floating-nav-brand {
    max-width: 3rem;
    margin-right: 0.5rem;
  }

  .alt-floating-nav-track {
    gap: 0.18rem;
    max-width: calc(100vw - 8.2rem);
  }

  .alt-floating-nav-btn {
    min-height: 1.9rem;
    padding: 0.34rem 0.58rem;
    font-size: 0.74rem;
  }

  .alt-floating-nav-meta {
    gap: 0.28rem;
  }

  .alt-floating-nav-status {
    min-height: 1.84rem;
    padding: 0.3rem 0.52rem;
    font-size: 0.64rem;
  }

  .alt-floating-nav-theme-toggle {
    min-width: 1.84rem;
    min-height: 1.84rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .alt-floating-nav,
  .alt-floating-nav-indicator,
  .alt-floating-nav-btn {
    transition: none !important;
  }
}
"""


_FLOATING_NAV_JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  if (!data) {
    return;
  }

  const root = parentElement.querySelector('#alt-floating-nav-root');
  if (!root) {
    return;
  }

  if (root.__cleanup) {
    root.__cleanup();
    root.__cleanup = null;
  }

  const items = Array.isArray(data.items) ? data.items : [];
  const currentPage = data.current_page || '';
  const status = data.status || {};
  const hideOnScroll = Boolean(data.hide_on_scroll);
  const scrollThreshold = Number(data.scroll_threshold || 72);
  const navLabel = data.aria_label || '페이지 탐색';
  const theme = data.theme || {};
  const themeMode = data.theme_mode || 'light';
  const themeToggleLabel = data.theme_toggle_label || '테마 전환';
  const themeToggleIcon = data.theme_toggle_icon || '';

  root.innerHTML = `
    <nav class="alt-floating-nav" role="navigation" aria-label="${navLabel}">
      <div class="alt-floating-nav-shell">
        <div class="alt-floating-nav-brand" aria-hidden="true">
          <span class="alt-floating-nav-brandmark">Alt</span>
        </div>
        <div class="alt-floating-nav-track" role="tablist" aria-label="${navLabel}">
          <span class="alt-floating-nav-indicator" aria-hidden="true"></span>
          ${items.map((item) => `
            <button
              type="button"
              class="alt-floating-nav-btn"
              data-page-key="${item.key}"
              aria-current="${item.key === currentPage ? 'page' : 'false'}"
            >
              ${item.icon ? `<span class="alt-floating-nav-icon" aria-hidden="true">${item.icon}</span>` : ''}
              <span>${item.label}</span>
            </button>
          `).join('')}
        </div>
        <div class="alt-floating-nav-meta">
          <button
            type="button"
            class="alt-floating-nav-theme-toggle"
            aria-label="${themeToggleLabel}"
            title="${themeToggleLabel}"
            data-theme-mode="${themeMode}"
          >${themeToggleIcon}</button>
          <span
            class="alt-floating-nav-status ${status.state || 'stopped'}"
            title="${status.full_label || status.label || 'Stopped'}"
          >${status.label || 'Stop'}</span>
        </div>
      </div>
    </nav>
  `;

  const nav = root.querySelector('.alt-floating-nav');
  const shell = root.querySelector('.alt-floating-nav-shell');
  const track = root.querySelector('.alt-floating-nav-track');
  const indicator = root.querySelector('.alt-floating-nav-indicator');
  const buttons = Array.from(root.querySelectorAll('.alt-floating-nav-btn'));
  const themeToggle = root.querySelector('.alt-floating-nav-theme-toggle');

  if (!nav || !shell || !track || !indicator || buttons.length === 0) {
    return;
  }

  const applyTheme = () => {
    if (theme.background) {
      shell.style.setProperty('--alt-nav-surface', theme.background);
    }
    if (theme.background_secondary) {
      shell.style.setProperty('--alt-nav-surface-2', theme.background_secondary);
    }
    if (theme.border) {
      shell.style.setProperty('--alt-nav-border', theme.border);
    }
    if (theme.text) {
      shell.style.setProperty('--alt-nav-text', theme.text);
    }
    if (theme.text_strong) {
      shell.style.setProperty('--alt-nav-text-strong', theme.text_strong);
    }
    if (theme.primary) {
      shell.style.setProperty('--alt-nav-primary', theme.primary);
    }
    if (theme.hover) {
      shell.style.setProperty('--alt-nav-hover', theme.hover);
    }
    if (theme.chip) {
      shell.style.setProperty('--alt-nav-chip', theme.chip);
    }
    if (theme.chip_hover) {
      shell.style.setProperty('--alt-nav-chip-hover', theme.chip_hover);
    }
    if (theme.shadow) {
      shell.style.setProperty('--alt-nav-shadow', theme.shadow);
    }
    if (theme.top_offset) {
      shell.style.setProperty('--alt-nav-top-offset', theme.top_offset);
    }
    if (theme.top_offset_mobile) {
      shell.style.setProperty('--alt-nav-top-offset-mobile', theme.top_offset_mobile);
    }
  };

  applyTheme();

  const updateIndicator = () => {
    const activeButton = buttons.find((button) => button.dataset.pageKey === currentPage) || buttons[0];
    if (!activeButton) {
      indicator.style.opacity = '0';
      return;
    }
    indicator.style.opacity = '1';
    indicator.style.width = `${activeButton.offsetWidth}px`;
    indicator.style.transform = `translateX(${activeButton.offsetLeft}px)`;
  };

  buttons.forEach((button) => {
    const pageKey = button.dataset.pageKey || '';
    button.addEventListener('click', () => {
      if (!pageKey || pageKey === currentPage) {
        return;
      }
      setTriggerValue('selected_page', pageKey);
    });
    button.addEventListener('keydown', (event) => {
      if ((event.key === 'Enter' || event.key === ' ') && pageKey && pageKey !== currentPage) {
        event.preventDefault();
        setTriggerValue('selected_page', pageKey);
      }
    });
  });

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      setTriggerValue('theme_toggle_event', `${Date.now()}`);
    });
  }

  let lastScrollY = window.scrollY;
  let ticking = false;
  let forcedVisible = false;
  let titleObserver = null;

  const setVisibility = (visible) => {
    nav.classList.toggle('is-hidden', !visible);
  };

  const evaluateVisibility = () => {
    const currentScrollY = window.scrollY;
    if (!hideOnScroll || currentScrollY <= scrollThreshold) {
      setVisibility(true);
    } else {
      const scrollingDown = currentScrollY > lastScrollY + 2;
      const scrollingUp = currentScrollY < lastScrollY - 2;
      if (forcedVisible || scrollingUp) {
        setVisibility(true);
      } else if (scrollingDown) {
        setVisibility(false);
      }
    }
    lastScrollY = currentScrollY;
    ticking = false;
  };

  const onScroll = () => {
    if (ticking) {
      return;
    }
    ticking = true;
    window.requestAnimationFrame(evaluateVisibility);
  };

  const onResize = () => {
    window.requestAnimationFrame(updateIndicator);
    window.requestAnimationFrame(evaluateVisibility);
  };

  const onMouseEnter = () => {
    forcedVisible = true;
    setVisibility(true);
  };

  const onMouseLeave = () => {
    forcedVisible = false;
    evaluateVisibility();
  };

  nav.addEventListener('mouseenter', onMouseEnter);
  nav.addEventListener('mouseleave', onMouseLeave);
  nav.addEventListener('focusin', onMouseEnter);
  nav.addEventListener('focusout', onMouseLeave);
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onResize, { passive: true });

  const bindTitleObserver = () => {
    const titleAnchor = document.getElementById('alt-page-title-anchor');
    if (!titleAnchor || typeof IntersectionObserver === 'undefined') {
      return;
    }
    titleObserver = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        nav.classList.toggle('show-brand', !entry.isIntersecting);
      },
      {
        root: null,
        threshold: 0.02,
        rootMargin: '-72px 0px 0px 0px',
      }
    );
    titleObserver.observe(titleAnchor);
  };

  updateIndicator();
  evaluateVisibility();
  window.requestAnimationFrame(bindTitleObserver);

  root.__cleanup = () => {
    window.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', onResize);
    nav.removeEventListener('mouseenter', onMouseEnter);
    nav.removeEventListener('mouseleave', onMouseLeave);
    nav.removeEventListener('focusin', onMouseEnter);
    nav.removeEventListener('focusout', onMouseLeave);
    if (titleObserver) {
      titleObserver.disconnect();
    }
  };
}
"""


_FLOATING_NAV_COMPONENT = None


def _get_component():
    global _FLOATING_NAV_COMPONENT
    if _FLOATING_NAV_COMPONENT is None:
        _FLOATING_NAV_COMPONENT = components_v2.component(
            "alt_floating_nav_v2",
            html=_FLOATING_NAV_HTML,
            css=_FLOATING_NAV_CSS,
            js=_FLOATING_NAV_JS,
        )
    return _FLOATING_NAV_COMPONENT


def theme_tokens(theme_mode: str) -> dict[str, str]:
    primary = st.get_option("theme.primaryColor") or "#3b82f6"
    background_color = st.get_option("theme.backgroundColor") or "#ffffff"
    secondary_background = st.get_option("theme.secondaryBackgroundColor") or "#f8fafc"
    text_color = st.get_option("theme.textColor") or "#111827"
    base = theme_mode if theme_mode in {"light", "dark"} else str(st.get_option("theme.base") or "light")
    if base == "dark":
        background = "rgba(12, 18, 32, 0.76)"
        background_secondary = "rgba(18, 25, 40, 0.72)"
        border = "rgba(226, 232, 240, 0.12)"
        text = "rgba(226, 232, 240, 0.86)"
        text_strong = "rgba(248, 250, 252, 0.98)"
        hover = "rgba(255, 255, 255, 0.08)"
        chip = "rgba(255, 255, 255, 0.06)"
        chip_hover = "rgba(255, 255, 255, 0.12)"
        shadow = "0 18px 42px rgba(2, 6, 23, 0.24), 0 6px 16px rgba(2, 6, 23, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.06)"
    else:
        background = _hex_to_rgba(background_color, 0.78, "rgba(255, 255, 255, 0.78)")
        background_secondary = _hex_to_rgba(secondary_background, 0.9, "rgba(248, 250, 252, 0.9)")
        border = _hex_to_rgba(text_color, 0.1, "rgba(15, 23, 42, 0.1)")
        text = _hex_to_rgba(text_color, 0.78, "rgba(15, 23, 42, 0.78)")
        text_strong = _hex_to_rgba(text_color, 0.96, "rgba(15, 23, 42, 0.96)")
        hover = _hex_to_rgba(primary, 0.1, "rgba(59, 130, 246, 0.1)")
        chip = _hex_to_rgba(text_color, 0.05, "rgba(15, 23, 42, 0.05)")
        chip_hover = _hex_to_rgba(primary, 0.14, "rgba(59, 130, 246, 0.14)")
        shadow = "0 18px 42px rgba(15, 23, 42, 0.08), 0 6px 16px rgba(15, 23, 42, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.72)"
    return {
        "base": base,
        "primary": primary,
        "background": background,
        "background_secondary": background_secondary,
        "border": border,
        "text": text,
        "text_strong": text_strong,
        "hover": hover,
        "chip": chip,
        "chip_hover": chip_hover,
        "shadow": shadow,
        "top_offset": "0.58rem",
        "top_offset_mobile": "0.42rem",
    }


def build_nav_items(items: Iterable[FloatingNavItem]) -> list[dict[str, str]]:
    return [{"key": item.key, "label": item.label, "icon": item.icon} for item in items]


def resolve_current_page_key(page_map: Mapping[str, object], selected_page: object, default_key: str) -> str:
    for key, page in page_map.items():
        if page is selected_page:
            return key
    return default_key


def render_floating_nav(
    *,
    current_page: str,
    items: Iterable[FloatingNavItem],
    status: Mapping[str, Any],
    theme_mode: str,
    hide_on_scroll: bool = True,
    scroll_threshold: int = 72,
    key: str = "alt-floating-nav",
) -> dict[str, str | None]:
    component = _get_component()
    result = component(
        key=key,
        data={
            "current_page": current_page,
            "items": build_nav_items(items),
            "status": {
                "state": str(status.get("state", "stopped")).lower(),
                "label": _toolbar_status_label(status),
                "full_label": str(status.get("label", "Stopped") or "Stopped"),
            },
            "theme": theme_tokens(theme_mode),
            "theme_mode": theme_mode,
            "theme_toggle_label": "라이트/다크 모드 전환",
            "theme_toggle_icon": _theme_toggle_svg(theme_mode),
            "hide_on_scroll": bool(hide_on_scroll),
            "scroll_threshold": int(scroll_threshold),
            "aria_label": "Alt 화면 탐색",
        },
        on_selected_page_change=lambda: None,
        on_theme_toggle_event_change=lambda: None,
    )
    selected_page = getattr(result, "selected_page", None)
    theme_toggle_event = getattr(result, "theme_toggle_event", None)
    return {
        "selected_page": str(selected_page) if selected_page else None,
        "theme_toggle_event": str(theme_toggle_event) if theme_toggle_event else None,
    }


def render_navigation_fallback(
    *,
    current_page: str,
    items: list[FloatingNavItem],
    page_map: Mapping[str, Any],
    segmented_key: str = "alt-nav-segmented",
) -> None:
    try:
        selected = st.segmented_control(
            "화면 전환",
            options=[item.key for item in items],
            default=current_page,
            format_func=lambda key: next((item.label for item in items if item.key == key), key),
            key=segmented_key,
        )
        if selected and selected != current_page:
            st.switch_page(page_map[selected])
        return
    except Exception:
        pass

    cols = st.columns(len(items))
    for idx, item in enumerate(items):
        with cols[idx]:
            st.page_link(page_map[item.key], label=item.label, icon=item.icon or None)
