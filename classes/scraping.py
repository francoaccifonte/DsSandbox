import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class LNBot:
    def __init__(self):
        self.driver = webdriver.Chrome('./support/chromedriver.exe')
        self.search_url = 'https://www.linkedin.com/jobs/'
    
    def JobSearch(self, query, place = None):
        self.driver.get(self.search_url)
        self.driver.find_element_by_xpath('//*[@id="JOBS"]/section[1]/input').send_keys(query)
        self.driver.find_element_by_xpath('//*[@id="JOBS"]/section[2]/input').clear()
        self.driver.find_element_by_xpath('//*[@id="JOBS"]/section[2]/input').send_keys(place + Keys.ENTER)
        self.search_result_ammount = self.driver.find_element_by_xpath('//*[@id="main-content"]/div/section/div[2]/h1/span[1]').text
        self.JobLinkList = []
        with open('./results/scraping/results.txt', 'w') as result_file:
            for ii in range(1, 20):
                url = self.driver.find_element_by_xpath('//*[@id="main-content"]/div/section/ul/li[' + str(ii) + ']/a').get_attribute('href')
                self.JobLinkList.append(url)
                result_file.write(url + '\n')
        