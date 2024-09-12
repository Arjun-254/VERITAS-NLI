from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time


def people_also_ask(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument('log-level=1')

    cService = Service(executable_path='PATH_TO_SELENIUM_WEBDRIVER')
    driver = webdriver.Chrome(service=cService,options=chrome_options)
    driver.get(url)
    try:
        wait = WebDriverWait(driver, 5)
        questions = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//div[@jsname='yEVEwb']")))
    except Exception as e:
        # print(f"People Also Asked questions dont exist for this category")
        driver.close()
        driver.quit()

        return []
    
    all_span_texts = []  
    
    for div in questions:
        try:
            div.click()
            time.sleep(1)
            
            span_elements = div.find_elements(By.TAG_NAME, 'span')
            for span in span_elements:
                text = span.text.strip()
                if text and text not in all_span_texts:
                    all_span_texts.append(text)
                    
        except Exception as e:
            print(f"Error clicking or extracting from div/span: {e}")
    driver.close()
    driver.quit()

    return all_span_texts[0:2]

