""" Download data sources """

import requests
import shutil
import sys, os


def extract_all(archives, extract_path):
    for filename in archives:
        shutil.unpack_archive(filename, extract_path)

def download_archive(url, target_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
        return True

    return False

if __name__ == "__main__":
    source_dir = "sources"
    os.makedirs(source_dir, exist_ok=True)

    if sys.argv[1] == "iamondo":
        target_path = os.path.join(source_dir, 'IAMonDo-db-1.0.tar.gz')
        url = f'https://fki.tic.heia-fr.ch/DBs/iamOnDoDB/data/{target_path}'
        
        download_archive(url, target_path)
        extract_all(target_path, source_dir)
    elif sys.argv[1] == "apub":
        print("Must be downloaded manually")
    elif sys.argv[1] == "prima":
        print("PRiMA must be downloaded manually")
    elif sys.argv[1] == "bdid":
        target_path = os.path.join(source_dir, 'BDID_Sample_BasicGT.zip')
        url = f'https://web.uvic.ca/~mcote/BDID/{target_path}'

