"""Real browser checks; use installed Chrome or `playwright install chromium`."""
import pytest

playwright = pytest.importorskip("playwright.sync_api")


def test_selection_submission_clear_and_mobile_layout(live_url, browser_page):
    page = browser_page
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(live_url)
    page.get_by_text("Browse all available names", exact=True).click()
    for name in ["itching", "skin_rash", "nodal_skin_eruptions"]:
        page.select_option("#symptom-picker", name)
        page.click("#add-symptom")
    assert page.input_value("#symptoms") == "itching, skin rash, nodal skin eruptions"
    page.get_by_role("button", name="Explore pattern").click()
    assert page.get_by_role("heading", name="Dataset match: Fungal infection").is_visible()
    page.click("#clear-symptoms")
    assert page.input_value("#symptoms") == ""
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert not errors


def test_voice_updates_submitted_field(live_url, browser_page):
    page = browser_page
    page.add_init_script("""
      window.SpeechRecognition = class {
        start() {
          this.onresult({results: [[{transcript: 'itching comma skin rash comma nodal skin eruptions'}]]});
          this.onend();
        }
      };
    """)
    page.goto(live_url)
    page.get_by_text("Optional voice input", exact=True).click()
    page.get_by_role("button", name="Use microphone").click()
    assert page.input_value("#symptoms") == "itching , skin rash , nodal skin eruptions"
    assert page.get_by_role("button", name="Use microphone").is_enabled()
    page.get_by_role("button", name="Explore pattern").click()
    assert page.get_by_role("heading", name="Dataset match: Fungal infection").is_visible()


def test_unsupported_speech_fallback_and_navigation(live_url, browser_page):
    page = browser_page
    page.add_init_script("window.SpeechRecognition = undefined; window.webkitSpeechRecognition = undefined;")
    page.goto(live_url)
    page.get_by_text("Optional voice input", exact=True).click()
    assert page.get_by_text("Voice input is unavailable", exact=False).is_visible()
    assert page.get_by_role("button", name="Use microphone").is_hidden()
    page.get_by_role("link", name="About", exact=True).click()
    assert page.get_by_role("heading", name="Learning from symptom patterns").is_visible()
    page.get_by_role("link", name="Explore", exact=True).click()
    assert page.get_by_role("heading", name="Explore symptom patterns").is_visible()


def test_search_deduplication_removal_and_empty_results(live_url, browser_page):
    page = browser_page
    page.goto(live_url)
    search = page.locator("#symptom-search")
    search.fill("skin rash")
    page.locator("#symptom-search-results [data-symptom='skin_rash']").click()
    assert page.locator("#selected-count").inner_text() == "1"
    page.locator(".quick-symptom[data-symptom='skin_rash']").click()
    assert page.locator("#selected-count").inner_text() == "1"
    page.get_by_role("button", name="Remove skin rash", exact=True).click()
    assert page.locator("#selected-count").inner_text() == "0"
    search.fill("not-a-real-symptom")
    assert page.locator("#no-search-results").is_visible()
    search.press("Escape")
    assert page.locator("#symptom-search-results").is_hidden()


def test_mobile_menu_and_evidence_pages(live_url, browser_page):
    page = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(live_url)
    page.get_by_role("button", name="Toggle navigation").click()
    page.get_by_role("link", name="Model & results", exact=True).click()
    assert page.get_by_role("heading", name="A model you can inspect.").is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.get_by_role("button", name="Toggle navigation").click()
    page.get_by_role("link", name="Data & sources", exact=True).click()
    assert page.get_by_role("heading", name="Transparency comes first.").is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_manual_entry_works_without_javascript(live_url, browser_page):
    context = browser_page.context.browser.new_context(java_script_enabled=False)
    page = context.new_page()
    try:
        page.goto(live_url)
        assert page.get_by_role("switch", name="Dark theme").is_hidden()
        page.fill("#symptoms", "itching,skin rash,nodal skin eruptions")
        page.get_by_role("button", name="Explore pattern").click()
        assert page.get_by_role("heading", name="Dataset match: Fungal infection").is_visible()
        page.get_by_role("button", name="Run Skin blemishes example").click()
        assert page.get_by_role("heading", name="Dataset match: Acne").is_visible()
        assert page.get_by_text("Fictional input", exact=False).is_visible()
    finally:
        context.close()


def test_example_buttons_scores_and_mobile_results(live_url, browser_page):
    page = browser_page
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(live_url)
    page.locator(".quick-symptom[data-symptom='headache']").click()
    page.get_by_role("button", name="Run Skin & itching example").click()
    assert page.get_by_role("heading", name="Dataset match: Fungal infection").is_visible()
    assert page.locator("#selected-count").inner_text() == "3"
    assert "headache" not in page.input_value("#symptoms")
    assert page.locator(".result-metrics").get_by_text("Model score", exact=True).is_visible()
    assert page.locator(".result-metrics").get_by_text("Held-out test accuracy", exact=True).is_visible()
    assert page.locator(".result-metrics > div").nth(1).get_by_text("100.0%", exact=True).is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.get_by_text("Enter names manually", exact=True).click()
    page.fill("#symptoms", "itching, ITCHING, itching")
    assert page.locator("#selected-count").inner_text() == "1"
    page.get_by_role("button", name="Explore pattern").click()
    assert page.get_by_role("heading", name="More symptoms needed").is_visible()
    assert page.get_by_text("You selected 1 unique symptom.", exact=False).is_visible()
    assert page.locator(".result-metrics").count() == 0


def test_theme_choice_survives_submission_navigation_and_reload(live_url, browser_page):
    page = browser_page
    page.emulate_media(color_scheme="light")
    page.goto(live_url)
    toggle = page.get_by_role("switch", name="Dark theme")
    playwright.expect(toggle).to_have_attribute("aria-checked", "false")
    light_surface = page.locator(".card").first.evaluate("el => getComputedStyle(el).backgroundColor")
    toggle.focus()
    page.keyboard.press("Space")
    playwright.expect(toggle).to_have_attribute("aria-checked", "true")
    assert page.evaluate("localStorage.getItem('healthcaredl-theme')") == "dark"
    assert page.locator(".card").first.evaluate("el => getComputedStyle(el).backgroundColor") != light_surface
    assert page.locator('meta[name="theme-color"]').get_attribute("content") == "#0e1922"
    page.get_by_role("button", name="Run Skin & itching example").click()
    assert page.get_by_role("heading", name="Dataset match: Fungal infection").is_visible()
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.get_by_role("link", name="Model & results", exact=True).click()
    page.reload()
    playwright.expect(page.get_by_role("switch", name="Dark theme")).to_have_attribute("aria-checked", "true")
    assert page.locator(".loss-chart").is_visible()
    page.get_by_role("switch", name="Dark theme").click()
    page.reload()
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert page.locator(".card").first.evaluate("el => getComputedStyle(el).backgroundColor") == light_surface


def test_system_theme_is_used_until_a_theme_is_chosen(live_url, browser_page):
    page = browser_page
    page.emulate_media(color_scheme="dark")
    page.goto(live_url)
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert page.evaluate("localStorage.getItem('healthcaredl-theme')") is None
    page.emulate_media(color_scheme="light")
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.get_by_role("switch", name="Dark theme").click()
    page.emulate_media(color_scheme="dark")
    page.emulate_media(color_scheme="light")
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "dark")


@pytest.mark.parametrize("storage", ["blocked", "invalid"])
def test_theme_tolerates_unavailable_or_invalid_storage(live_url, browser_page, storage):
    page = browser_page
    page.emulate_media(color_scheme="light")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    if storage == "blocked":
        page.add_init_script("Object.defineProperty(window, 'localStorage', {get() {throw new DOMException('Blocked', 'SecurityError');}});")
    else:
        page.add_init_script("localStorage.setItem('healthcaredl-theme', 'invalid-value');")
    page.goto(live_url)
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.get_by_role("switch", name="Dark theme").click()
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.get_by_role("button", name="Run Skin blemishes example").click()
    assert page.get_by_role("heading", name="Dataset match: Acne").is_visible()
    assert not errors


def test_mobile_theme_switch_stays_visible_and_syncs_tabs(live_url, browser_page):
    page = browser_page
    page.emulate_media(color_scheme="light")
    page.set_viewport_size({"width": 320, "height": 812})
    page.goto(live_url)
    assert page.get_by_role("switch", name="Dark theme").is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    second = page.context.new_page()
    try:
        second.emulate_media(color_scheme="light")
        second.goto(live_url + "/privacy")
        page.get_by_role("switch", name="Dark theme").click()
        playwright.expect(second.locator("html")).to_have_attribute("data-theme", "dark")
        second.evaluate("localStorage.removeItem('healthcaredl-theme')")
        playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
        assert page.get_by_role("button", name="Toggle navigation").is_visible()
    finally:
        second.close()
