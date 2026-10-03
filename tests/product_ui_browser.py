from __future__ import annotations

import contextlib
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "product-client" / "web"
EVIDENCE = ROOT / "artifacts" / "product-ui"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


@contextlib.contextmanager
def static_server():
    handler = lambda *args, **kwargs: QuietHandler(
        *args,
        directory=str(WEB),
        **kwargs,
    )
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            thread.join(timeout=2)


def assert_min_target(locator, minimum=44):
    box = locator.bounding_box()
    assert box is not None
    assert box["width"] >= minimum, box
    assert box["height"] >= minimum, box


def run_desktop(browser, base_url):
    context = browser.new_context(
        viewport={"width": 1280, "height": 820},
        locale="vi-VN",
    )
    page = context.new_page()
    page.goto(base_url, wait_until="networkidle")

    power = page.locator("#powerButton")
    composer = page.locator("#messageInput")
    send = page.locator("#sendButton")

    assert power.get_attribute("role") == "switch"
    assert power.get_attribute("aria-checked") == "false"
    assert composer.is_disabled()
    assert_min_target(power)
    assert_min_target(send)

    power.click()
    page.wait_for_function(
        "() => document.querySelector('#powerButton').dataset.phase === 'on'"
    )
    assert power.get_attribute("aria-checked") == "true"
    assert not composer.is_disabled()

    composer.fill("Chào Nolane")
    send.click()
    page.wait_for_selector(".message.user")
    page.wait_for_selector(".message.assistant")
    assert page.locator(".message.user").last.inner_text() == "Chào Nolane"
    assert "runtime Nolane thật" in page.locator(".message.assistant").last.inner_text()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(EVIDENCE / "desktop-chat.png"), full_page=True)

    page.locator("#profileButton").click()
    dialog = page.locator("#personalizationDialog")
    assert dialog.evaluate("(el) => el.open")
    page.screenshot(
        path=str(EVIDENCE / "desktop-personalization.png"),
        full_page=True,
    )
    page.locator("#preferredName").fill("Tài")
    memory = page.locator("#memoryEnabled")
    assert memory.is_checked()
    page.locator(".toggle-row").click()
    assert not memory.is_checked()
    page.locator("#saveProfileButton").click()
    page.wait_for_function(
        "() => !document.querySelector('#personalizationDialog').open"
    )

    page.locator("#profileButton").click()
    assert page.locator("#preferredName").input_value() == "Tài"
    assert not page.locator("#memoryEnabled").is_checked()
    page.keyboard.press("Escape")
    assert not dialog.evaluate("(el) => el.open")

    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1
    page.close()


def run_mobile(browser, base_url):
    context = browser.new_context(
        viewport={"width": 390, "height": 844},
        is_mobile=True,
        has_touch=True,
        device_scale_factor=2,
        locale="vi-VN",
    )
    page = context.new_page()
    page.goto(base_url, wait_until="networkidle")

    power = page.locator("#powerButton")
    assert_min_target(power)
    power.tap()
    page.wait_for_function(
        "() => document.querySelector('#powerButton').dataset.phase === 'on'"
    )

    page.locator("#profileButton").tap()
    dialog = page.locator("#personalizationDialog")
    assert dialog.evaluate("(el) => el.open")
    box = dialog.bounding_box()
    assert box is not None
    assert box["width"] >= 388

    page.screenshot(
        path=str(EVIDENCE / "mobile-personalization.png"),
        full_page=True,
    )
    page.locator("#closeProfileButton").tap()
    assert not dialog.evaluate("(el) => el.open")
    page.screenshot(path=str(EVIDENCE / "mobile-chat.png"), full_page=True)

    composer = page.locator("#messageInput")
    composer.tap()
    composer.fill("Xin chào")
    assert page.locator("#sendButton").is_enabled()
    assert_min_target(page.locator("#sendButton"))

    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1
    context.close()


def run_reduced_motion(browser, base_url):
    context = browser.new_context(reduced_motion="reduce", locale="vi-VN")
    page = context.new_page()
    page.goto(base_url, wait_until="networkidle")
    page.locator("#powerButton").click()
    page.wait_for_function(
        "() => document.querySelector('#powerButton').dataset.phase === 'on'"
    )
    animation = page.locator(".ember-core").evaluate(
        "(el) => getComputedStyle(el).animationDuration"
    )
    assert animation in {"0.001s", "1ms", "0s"}, animation
    context.close()


def main() -> int:
    if not (WEB / "index.html").is_file():
        raise SystemExit("product web entry is missing")
    with static_server() as base_url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                run_desktop(browser, base_url)
                run_mobile(browser, base_url)
                run_reduced_motion(browser, base_url)
            finally:
                browser.close()
    print("NUI product browser court: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
