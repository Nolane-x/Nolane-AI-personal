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

    language = page.locator("#uiLanguage")
    assert language.input_value() == "en"
    assert page.locator("#messageInput").get_attribute("placeholder") == "Message Nolane…"
    language.select_option("vi")
    assert page.locator("#messageInput").get_attribute("placeholder") == "Nhắn cho Nolane…"
    page.reload(wait_until="networkidle")
    assert page.locator("#uiLanguage").input_value() == "vi"
    assert page.locator("#messageInput").get_attribute("placeholder") == "Nhắn cho Nolane…"

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

    mind_button = page.locator("#mindStateButton")
    assert mind_button.is_visible()
    mind_button.click()
    mind_dialog = page.locator("#mindDialog")
    assert mind_dialog.evaluate("(el) => el.open")
    assert page.locator("#mindMood").inner_text()
    assert page.locator("#mindActivity").inner_text()
    assert page.locator("#emotionChips .emotion-chip").count() >= 1
    page.screenshot(path=str(EVIDENCE / "desktop-mind-panel.png"), full_page=True)
    page.locator("#closeMindButton").click()
    assert not mind_dialog.evaluate("(el) => el.open")

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

    learning = page.locator("#learningDetails")
    learning.locator("summary").click()
    assert learning.evaluate("(el) => el.open")
    assert page.locator("#openLearningReview").is_disabled()

    page.locator("#prepareLearningWindow").click()
    learning_dialog = page.locator("#learningDialog")
    page.wait_for_function(
        "() => document.querySelector('#learningDialog').open"
    )
    assert learning_dialog.evaluate("(el) => el.open")
    assert page.locator("#reviewCandidate").is_visible()
    assert page.locator("#reviewPrompt").inner_text()
    assert page.locator("#reviewTarget").input_value()
    assert page.locator("#finalizeLearning").is_hidden()
    page.screenshot(
        path=str(EVIDENCE / "desktop-learning-review.png"),
        full_page=True,
    )

    corrected_target = "Câu trả lời đã được người dùng sửa trước khi Nolane học."
    page.locator("#reviewTarget").fill(corrected_target)
    assert page.locator("#reviewTarget").input_value() == corrected_target
    page.locator("#approveLearning").click()
    page.wait_for_function(
        "() => document.querySelector('#learningProgress').textContent.startsWith('1/3')"
    )
    page.locator("#rejectLearning").click()
    page.wait_for_function(
        "() => document.querySelector('#learningProgress').textContent.startsWith('2/3')"
    )
    page.locator("#markSensitive").click()
    page.wait_for_function(
        "() => !document.querySelector('#finalizeLearning').hidden"
    )
    page.screenshot(
        path=str(EVIDENCE / "desktop-learning-completion-state.png"),
        full_page=True,
    )
    assert learning_dialog.evaluate("(el) => el.open")
    assert page.locator("#reviewComplete").is_visible()
    assert page.locator("#reviewActions").is_hidden()
    assert page.locator("#finalizeLearning").is_visible()
    page.screenshot(
        path=str(EVIDENCE / "desktop-learning-complete.png"),
        full_page=True,
    )

    page.locator("#finalizeLearning").click()
    page.wait_for_function(
        "() => !document.querySelector('#learningDialog').open"
    )
    assert dialog.evaluate("(el) => el.open")
    assert "1 " in page.locator("#learningSummary").inner_text()
    assert page.locator("#openLearningReview").is_disabled()

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

    assert page.locator("#uiLanguage").input_value() == "en"
    page.locator("#uiLanguage").select_option("vi")
    assert page.locator("#messageInput").get_attribute("placeholder") == "Nhắn cho Nolane…"

    power = page.locator("#powerButton")
    assert_min_target(power)
    power.tap()
    page.wait_for_function(
        "() => document.querySelector('#powerButton').dataset.phase === 'on'"
    )

    mind_button = page.locator("#mindStateButton")
    assert mind_button.is_visible()
    mind_button.tap()
    mind_dialog = page.locator("#mindDialog")
    assert mind_dialog.evaluate("(el) => el.open")
    page.screenshot(path=str(EVIDENCE / "mobile-mind-panel.png"), full_page=True)
    page.locator("#closeMindButton").tap()
    assert not mind_dialog.evaluate("(el) => el.open")

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
