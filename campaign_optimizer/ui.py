"""Componentes HTML del panel: tarjetas KPI, paneles, barras, calendario y tablas."""

from __future__ import annotations

from calendar import Calendar
from html import escape

TONES = {"ok": "rc-badge-ok", "warn": "rc-badge-warn", "bad": "rc-badge-bad"}


def esc(value: object) -> str:
    return escape(str(value))


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="rc-badge {TONES.get(tone, "rc-badge-neutral")}">{esc(text)}</span>'


def chip(text: str) -> str:
    return f'<span class="rc-chip">{esc(text)}</span>'


def toolbar(title: str, subtitle: str = "", right: str = "") -> str:
    right_html = f'<div class="rc-toolbar-right">{right}</div>' if right else ""
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return f'<div class="rc-toolbar"><div><h1>{esc(title)}</h1>{sub}</div>{right_html}</div>'


def kpis(items: list[dict], highlight: int | None = None) -> str:
    cards = []
    for index, item in enumerate(items):
        cls = "rc-kpi rc-kpi-hl" if index == highlight else "rc-kpi"
        note = f'<div class="rc-kpi-note">{esc(item["note"])}</div>' if item.get("note") else ""
        cards.append(
            f'<div class="{cls}"><div class="rc-kpi-value">{item["value"]}</div>'
            f'<div class="rc-kpi-label">{esc(item["label"])}</div>{note}</div>'
        )
    return f'<div class="rc-kpis">{"".join(cards)}</div>'


def panel(title: str, body: str, sub: str = "") -> str:
    sub_html = f'<span class="rc-panel-sub">{esc(sub)}</span>' if sub else ""
    return (
        '<div class="rc-panel">'
        f'<div class="rc-panel-head"><span class="rc-panel-title">{esc(title)}</span>{sub_html}</div>'
        f'<div class="rc-panel-body">{body}</div></div>'
    )


def group(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="rc-row"><span class="rc-row-label">{esc(label)}</span>'
        f'<span class="rc-row-value">{value}</span></div>'
        for label, value in rows
    )
    return f'<div class="rc-group">{body}</div>'


def table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{"num" if i in numeric else ""}">{cell}</td>' for i, cell in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="rc-table-wrap"><table class="rc-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def bars(rows: list[tuple[str, str, float]]) -> str:
    """Barras de progreso: etiqueta, valor y porcentaje (0-100)."""
    items = []
    for label, value, pct in rows:
        width = max(0.0, min(100.0, pct))
        items.append(
            '<div class="rc-bar-row">'
            f'<span class="rc-bar-label">{esc(label)}</span>'
            f'<span class="rc-bar-value">{value}</span>'
            f'<span class="rc-bar-track"><span class="rc-bar-fill" style="width:{width:.1f}%"></span></span>'
            "</div>"
        )
    return f'<div class="rc-bars">{"".join(items)}</div>'


def month_grid(year: int, month: int, active_days: set[int]) -> str:
    """Calendario del periodo con los días que tienen datos resaltados."""
    weeks = Calendar(firstweekday=0).monthdayscalendar(year, month)
    dow = "".join(f'<div class="rc-cal-dow">{d}</div>' for d in ("L", "M", "X", "J", "V", "S", "D"))
    cells = []
    for week in weeks:
        for day in week:
            if day == 0:
                cells.append('<div class="rc-cal-day is-empty"></div>')
            else:
                cls = "rc-cal-day is-on" if day in active_days else "rc-cal-day"
                cells.append(f'<div class="{cls}">{day}</div>')
    meses = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre")
    label = f"{meses[month - 1]} {year}"
    return (
        f'<div class="rc-panel-sub" style="margin-bottom:8px">{label} {year}</div>'
        f'<div class="rc-cal">{dow}{"".join(cells)}</div>'
    )
