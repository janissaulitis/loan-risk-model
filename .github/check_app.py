"""Opens the live app in a real browser, wakes it if asleep, and (mode "test") clicks through every tab,
saving screenshots and page text so problems can be checked without a person opening it."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://loan-risk-model.streamlit.app/"
mode = sys.argv[1] if len(sys.argv) > 1 else "wake"
out = Path("out"); out.mkdir(exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1366, "height": 900}, color_scheme="dark")
    page.goto(URL, timeout=120_000)
    page.wait_for_timeout(8_000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        print("was asleep, waking up"); wake.click(); page.wait_for_timeout(90_000)
    app = page.frame_locator("iframe[title='streamlitApp']")
    app.locator(".k-name, h1").first.wait_for(timeout=240_000)
    print("awake:", app.locator(".k-name, h1").first.inner_text())
    if mode == "test":
        page.wait_for_timeout(5_000)
        errors = app.locator("[data-testid=stException]")
        for i, tab in enumerate(["F1 Call list", "F2 Try an application", "F3 How good", "F4 Business case", "F5 Fairness"]):
            app.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(2_500)
            page.screenshot(path=str(out / f"{i}_{tab.split()[1].lower()}.png"))
            print(f"{tab}: {errors.count()} errors")
        (out / "page.txt").write_text(app.locator("[data-testid=stMain]").inner_text())
        if errors.count():
            print("ERROR:", errors.first.inner_text()[:800]); sys.exit(1)
    b.close()
