"""Render a captured terminal session (text file) as a terminal-style PNG for the report (Playwright + Chromium).

    python render_terminal.py session.txt out.png
"""
import html
import pathlib
import sys

from playwright.sync_api import sync_playwright

src, out = sys.argv[1], sys.argv[2]
width = 1100
text = pathlib.Path(src).read_text()
doc = f"""<html><body style="margin:0;background:#1e1e1e">
<pre style="margin:0;padding:14px 18px;color:#e6e6e6;font:13px/1.45 'DejaVu Sans Mono',monospace;white-space:pre-wrap;width:{width}px">{html.escape(text)}</pre></body></html>"""
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": width + 36, "height": 200}, device_scale_factor=2)
    pg.set_content(doc)
    pg.locator("pre").screenshot(path=out)
    b.close()
print(out)
