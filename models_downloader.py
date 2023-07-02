import os
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List

from bs4 import BeautifulSoup
from retrying import retry

ssl._create_default_https_context = ssl._create_unverified_context

class ModelDownloader:
    def __init__(self, repo_id=None, worker_num: int=16) -> None:
        self.repo_id = repo_id
        self.links = []
        self.html = None
        self.executor = ThreadPoolExecutor(max_workers=worker_num)


    def get_repo_main_page(self, repo_id: str=None) -> str:
        if self.repo_id:
            repo_id = self.repo_id
        if not repo_id:
            raise ValueError("must specific a repo id")
        
        main_page_url = f"https://huggingface.co/{repo_id}/tree/main"
        print(f"[INFO] exec {main_page_url}...")
        resp = urllib.request.urlopen(main_page_url)
        self.html = resp.read().decode("utf-8")
        return self.html
    

    def get_download_links(self, html: str=None) -> List[str]:
        if self.html:
            html = self.html
        if not html:
            raise ValueError("must specific a repo id")
        parser = BeautifulSoup(html, features="html.parser")
        all_sources = parser.find_all("a", title="Download file")
        for source in all_sources:
            uri = source.get("href")
            if ".gitattributes" in uri:
                continue
            self.links.append(f"https://huggingface.co{uri}")
        return self.links 
    

    @retry(wait_random_min=100, wait_random_max=500)
    def download(self, link: str):
        print("trying...")
        flag = os.system(f"wget -N {link} -P tmp/")
        if flag != 0:
            print("wget failed. retrying...")
            raise ValueError("wrong flag")

    
    def batch_download(self):
        return [self.executor.submit(self.download, (link)) for link in self.links]
    
    def ytl(self):
        self.get_repo_main_page()
        self.get_download_links()
        self.batch_download()


if __name__ == "__main__":
    md = ModelDownloader("THUDM/chatglm2-6b")
    md.ytl()

